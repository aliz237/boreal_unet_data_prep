library(dplyr)
library(optparse)
library(stringr)
library(sf)

RH_columns <- function(gdf){
    return(names(gdf)[grep('^RH_[0-9]{2}$', names(gdf))])
}

rh_columns <- function(gdf){
    return(names(gdf)[grep('^rh[0-9]{2}$', names(gdf))])
}

rename_height_columns_to_pretrained_models <- function(gdf){
    return(
        gdf |>
        rename_with(~gsub('\\brh([0-9]{2})\\b', 'RH_\\1', .x), matches='^rh[0-9]{2}$') |>
        rename(RH_98=h_canopy)
    )
}

offset_RH_columns <- function(gdf, offset){
    RH_columns <- RH_columns(gdf)
    return(gdf |> mutate(across(all_of(RH_columns), ~. + offset)))
}

set_AGB_model_id <- function(gdf){
  return(
      gdf |>
      mutate(
          model_id = case_when(
              segment_landcover %in% c(111, 113, 121, 123) ~ "m3", # needle leaf
              segment_landcover %in% c(112, 114, 122, 124) ~ "m1", # broad leaf
              TRUE ~ "m8"
          )
      )
  )
}

reformat_for_AGB_modeling <- function(gdf, offset){
    gdf <- rename_height_columns_to_pretrained_models(gdf)
    gdf$h_canopy <- gdf$RH_98
    gdf <- offset_RH_columns(gdf, offset)
    gdf <- set_AGB_model_id(gdf)
    return(gdf)
}

GEDI2AT08AGB <- function(biomass_models, df, randomize=FALSE, max_n=Inf, sample=FALSE){
  if (sample && nrow(df) > max_n)
    df <- reduce_sample_size(df, max_n)

  df$AGB <- NA
  df$SE <- NA

  ids<-unique(df$model_id)
  n_models <- length(ids)

  for (i in ids){
    model_i <- biomass_models[[i]]

    # Modify coeffients through sampling variance covariance matrix
    if(randomize)
      model_i <- randomize_AGB_model(model_i)

    # Predict AGB and SE
    df$AGB[df$model_id==i] <- predict(model_i, newdata=df[df$model_id==i,])
    df$SE[df$model_id==i] <- summary(model_i)$sigma^2

    df$AGB[df$AGB < 0] <- 0.0

    # Calculate Correction Factor C
    C <- mean(model_i$model$`sqrt(AGBD)`^2) / mean(model_i$fitted.values^2)

    # Bias correction in case there is a systematic over or under estimation in the model
    df$AGB[df$model_id==i] <- C*(df$AGB[df$model_id==i]^2)
  }

  # Apply slopemask, validmask and landcover masks
  bad_lc <- c(0, 60, 80, 200, 50, 70)
  df$AGB[df$slopemask == 0 |
                df$ValidMask == 0 |
                df$segment_landcover %in% bad_lc] <- 0.0
  return(df)
}

DOY_and_solar_filter <- function(gdf, start_DOY, end_DOY, solar_elevation){
    filter <- which((gdf$doy >= start_DOY) &
                    (gdf$doy <= end_DOY) &
                    (gdf$solar_elevation < solar_elevation)
                  )
    return(filter)
}

reduce_sample_size <- function(df, sample_size){
    return(df[sample(row.names(df), sample_size, replace=FALSE),])
}

remove_stale_columns <- function(df, column_names) {
    columns_to_remove <- intersect(names(df), column_names)
    df <- df[, !names(df) %in% columns_to_remove, drop=FALSE]
    return(df)
}

remove_height_outliers <- function(gdf){
    # outliers based on +/- 3*sd away from the landcover mean height
    return(
        gdf |>
        filter(!is.na(h_canopy)) |>
        group_by(segment_landcover) |>
        mutate(
            thresh_hi=mean(h_canopy, na.rm=T) + 3 * sd(h_canopy, na.rm=T),
            thresh_lo=mean(h_canopy, na.rm=T) - 3 * sd(h_canopy, na.rm=T),
        ) |>
        ungroup() |>
        filter(h_canopy <= thresh_hi, h_canopy >=thresh_lo, h_canopy >=0)
    )
}

set_short_veg_height_to_zero <- function(df, slope_thresh){
    height_columns <- c(rh_columns(df), "h_canopy", "h_mean_canopy", 
                        "h_min_canopy", "h_max_canopy")
    cond <- df$segment_landcover == 100
    cond <- cond | ((df$segment_landcover %in% c(20, 30, 60, 100)) & 
                    (df$slope > slope_thresh))
    cond[is.na(cond)] <- FALSE
    df[cond, height_columns] <- 0.0
    return(df)
}

short_veg_filter <- function(df, slope_thresh){
    cond <- ((df$segment_landcover %in% c(20, 30, 60, 100)) & 
             (df$slope > slope_thresh))
    cond[is.na(cond)] <- FALSE
    df <- df[!cond, ,drop=FALSE]
    return(df)
}


filter_atl08 <- function(gdf, min_n=5000, minDOY=121, maxDOY=273, max_sol_el=5){
    night_time_in_season <- DOY_and_solar_filter(gdf, minDOY, maxDOY, 0)

    cat('train data size before any filtering:', nrow(gdf), '\n')
    cat('length(night_time_in_season):', length(night_time_in_season), '\n')

    if (length(night_time_in_season) < min_n){
        cat('number of samples is <', min_n, '\n')
        cat('Increasing max solar elevation to:', max_sol_el, '\n')
        
        night_time_in_season <- DOY_and_solar_filter(gdf, minDOY, maxDOY, max_sol_el)
        
        cat('length(night_time_in_season):', length(night_time_in_season), '\n')
    }
    
    print('Applying night time in season filter')
    gdf <- gdf[night_time_in_season, ]
    
    cat('training data size after filtering:', nrow(gdf), '\n')
    gdf <- remove_stale_columns(gdf, c("binsize", "num_bins"))
    
    return(gdf)
}


ICESat2_AGB_RH <- function(
    atl08_path, out_fn, biomass_models_path, minDOY=121, maxDOY=273, max_sol_el=5,
    min_n=5000, offset=100, zero_short_veg=FALSE, slope_thresh=Inf, max_n=Inf
){
    biomass_models <- get_biomass_models(biomass_models_path)
    gdf <- st_read(atl08_path)
    print(summary(gdf[c('doy', 'solar_elevation')]))
    gdf <- filter_atl08(gdf, min_n, minDOY, maxDOY, max_sol_el)

    if (zero_short_veg)
        gdf <- set_short_veg_height_to_zero(gdf, slope_thresh)

    needed_cols <- c('h_canopy', 'segment_landcover', 'slopemask', 'ValidMask',
                     rh_columns(gdf))

    gdf <- gdf |> select(all_of(needed_cols))
    gdf <- reformat_for_AGB_modeling(gdf, offset)
    gdf <- remove_height_outliers(gdf)
    cat('Shape after removing height outliers:', nrow(gdf), '\n')

    gdf <- gdf |> filter(if_all(-geometry, ~ !is.na(.x) & .x != -9999))
    cat('Shape after removing NAs:', dim(gdf), '\n')
    str(gdf)

    gdf <- GEDI2AT08AGB(biomass_models, gdf)
    gdf <- offset_RH_columns(gdf, -1*offset)
    out_cols = c('h_canopy', 'AGB', 'segment_landcover', RH_columns(gdf))
    st_write(gdf[out_cols], out_fn, driver='Parquet')
    return(gdf)
}

get_biomass_models <- function(biomass_models_path){
    base_dir <- dirname(biomass_models_path)
    untar(biomass_models_path, exdir=base_dir)
    biomass_model_fns <- list.files(path=base_dir, pattern='*.rds', full.names=TRUE)
    biomass_models <- lapply(biomass_model_fns, readRDS)
    names(biomass_models) <- paste0("m",1:length(biomass_models))
    print(biomass_models)
    return(biomass_models)
}

option_list <- list(
  make_option(
    c("-a", "--atl08_path"), type = "character",
    help = "Path to the atl08 parquet"
  ),
  make_option(
    c("-b", "--biomass_models_path"), type = "character",
    help = "Path to biomass model RDS tarball"
  ),
  make_option(
    c("-o", "--out_fn"), type = "character",
    help = "output file name"
  ),
  make_option(
    c("--help"), action = "store_true",
    help = "Show help message"
  )
)

opt_parser <- OptionParser(option_list = option_list, add_help_option = FALSE)
opt <- parse_args(opt_parser)

cat("Parsed arguments:\n")
print(opt)
if (!is.null(opt$help)) {
  print_help(opt_parser)
}

do.call(ICESat2_AGB_RH, opt)
