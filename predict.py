import numpy sa np
from pathlib import Path
import rasterio
from rasterio.windows import Window
from data_prep import gapfill, resample_topo_if_needed

def predict_raster(hls_path, topo_path, out_raster_path, model, patch_size=128, ndval=-9999, batch_size=64):
    batch = []
    ulxy = []
    topo_path = resample_topo_if_needed(Path(topo_path))
    topo = rasterio.open(topo_path)

    step_size = 100
    hls_patches_dropped = 0
    topo_patches_dropped = 0    
    ax = np.clip(np.minimum(np.linspace(0, 1, patch_size), np.linspace(1, 0, patch_size)) * 5, 0.01, 1)
    kernel = np.outer(ax, ax)

    with rasterio.open(hls_path) as hls:
        w, h, c = hls.width, hls.height, hls.count
        out_arr = np.full((h, w), 0, dtype=np.float32)
        count_arr = np.full((h, w), 0, dtype=np.float32)
        meta = hls.meta.copy()

        for j in list(range(0, w, step_size))[:-1] + [w-patch_size]:
            for i in list(range(0, h, step_size))[:-1] + [h-patch_size]:

                win = Window(j, i, patch_size, patch_size)
                hls_arr = hls.read(window=win).astype(np.float32)

                # can't have a null patch
                hls_arr[hls_arr == ndval] = np.nan
                if np.any(np.isnan(hls_arr).all(axis=(1,2))):
                    continue
                
                # fill NA
                if not gapfill(hls_arr, na_thresh=0.60):
                    hls_patches_dropped += 1
                    continue
                # same for topo
                slope = topo.read([2], window=win).astype(np.float32)
                slope[slope == ndval] = np.nan
                slope = np.clip(slope, 0, 90)
                slope /= 90.0
                if np.isnan(slope).all():
                    continue
                if not gapfill(slope, na_thresh=0.60):
                    topo_patches_dropped += 1
                    continue

                X = np.concatenate([hls_arr, slope])
                batch.append(np.moveaxis(X, 0, -1))
                ulxy.append((j, i))

                if len(batch) > batch_size:
                    preds = model.predict(np.array(batch))
                    for (x, y), pred in zip(ulxy, preds):
                        out_arr[y:y+patch_size, x:x+patch_size] += pred[:,:,0] * kernel
                        count_arr[y:y+patch_size, x:x+patch_size] += kernel
                    batch = []
                    ulxy = []
    
    if batch:
        preds = model.predict(np.array(batch))
        for (x, y), pred in zip(ulxy, preds):
            out_arr[y:y+patch_size, x:x+patch_size] += pred[:,:,0] * kernel
            count_arr[y:y+patch_size, x:x+patch_size] += kernel
        batch = []
        ulxy = []
    # out_arr[count_arr == 0] = ndval
    # out_arr[count_arr != 0] /= count_arr
    out_arr = np.divide(out_arr, count_arr, where=count_arr != 0)
    out_arr[count_arr == 0] = ndval
    print(f'min={np.min(out_arr)}, min={np.min(out_arr[out_arr != ndval])}')
    print(f"""{hls_patches_dropped} hls_patches_dropped,
    {topo_patches_dropped} topo_patches_dropped"""
    )
    meta.update({'count': 1, 'nodata': ndval, 'dtype': 'float32'})
    with rasterio.open(out_raster_path, 'w', **meta) as o:
        o.write(out_arr, 1)
        o.set_band_description(1, 'Ht')

    topo.close()
