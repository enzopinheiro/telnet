#!/usr/bin/bash

# Shell script to generate forecasts for a specific initialization date using a trained TelNet.
# Expect $1 as YYYYMM, compute init_date = 12 months earlier (YYYYMM) and $2 as configuration number.
if [ -z "$1" ]; then
    echo "Usage: $0 requires initialization date in the format YYYYMM as argument. Ex: To generate a forecast initialized in October of 2025 using configuration 1, use $0 202510 1" >&2
    exit 1
fi

# Try GNU date; fallback to arithmetic if not available
if date -d "${1}01" >/dev/null 2>&1; then
    init_date=$(date -d "${1}01 -12 months" +%Y%m)
    final_date=$(date -d "${1}01 -1 months" +%Y%m)
else
    year=${1:0:4}
    month=${1:4:2}
    if ! [[ $year =~ ^[0-9]{4}$ && $month =~ ^[0-9]{2}$ ]]; then
        echo "Invalid date format: $1 (expected YYYYMM)" >&2
        exit 1
    fi
    year=$((year-1))
    init_date=$(printf "%04d%02d" "$year" "$month")
    final_date=$(printf "%04d%02d" "$year" "$month")
fi

# Download ERSST and ERA5 data
python -W ignore download_preprocess_data.py -idate 202501 -fdate $final_date

# Update climate indices files
python -W ignore compute_climate_indices.py -fdate $final_date

# Generate forecasts
python -W ignore generate_forecast.py -fdate $final_date -c $2