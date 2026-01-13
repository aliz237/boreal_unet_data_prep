#!/usr/bin/env -S bash --login
set -euo pipefail
# This script is used to install any custom packages required by the algorithm.

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )

conda env create -f ${basedir}/environment.yml
