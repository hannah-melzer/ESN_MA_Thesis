#!/bin/bash

# Source directory containing the .nc files
input_dir="/data/Hannah/data/"
# Output directory where chunked files will be saved
output_dir="/data/Hannah/Kuro_new_2/"

# Iterate over all .nc files in the input directory
for nc_file in $input_dir*.nc; do
  # Extract the filename (without the path)
  filename=$(basename $nc_file)
  
  # Generate a sensible name for the chunked file (you can adjust this part)
  output_file="${output_dir}chunked_${filename}"

  # Run ncks command to chunk the data
  ncks -4 -O \
    -d nlat,1436,1691 -d nlon,2300,2811 \
    --cnk_dmn nlat,50 --cnk_dmn nlon,50 \
    $nc_file $output_file

  # Print confirmation message
  echo "Processed $filename and saved as $output_file"
done
