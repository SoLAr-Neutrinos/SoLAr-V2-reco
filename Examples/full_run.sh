#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Get the name of the parent directory
PARENT_DIR=$(basename "$SCRIPT_DIR")

# Check if the parent directory is "Examples"
if [ "$PARENT_DIR" == "Examples" ]; then
  # Save the current directory and change to the parent directory
  pushd "$SCRIPT_DIR/.."
fi

# Full analysis
for light_file in /eos/experiment/dune/solar/data/SoLAr_v2/Light/root/46v_12db_th950_deco/deco_v6*
do
  # Print the current file name
  echo Light file: "$light_file"

  # Extract the timestamp after 'hit_' and before '.root' using sed to get the output folder
  label=$(echo $light_file | sed -n 's/.*_\([0-9]\{8\}_[0-9]\{6\}\)\.data\.root/\1/p')
  
  # Transform the label to the charge file timestamp format
  formatted_date=$(echo $label | sed 's/\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)_\([0-9]\{2\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)/\1_\2_\3_\4_\5/')

  charge_file="/eos/experiment/dune/solar/data/SoLAr_v2/Charge/cosmics/root/evd_self_trigger-packets-${formatted_date}_CEST_validated.root"
  echo Charge file: "$charge_file"

  # if [ -d "/eos/experiment/dune/solar/scripts/SoLAr-V2-reco/$label" ]; then
  #   continue
  # fi

  # Run the reconstruction script again without simulating dead areas
  python -m solarv2 reco -c "$charge_file" -l "$light_file" 

  python -m solarv2 display "$label" -s -n
  
  # Run the analysis script again on the output folder
  python -m solarv2 ana "$label"
  python -m solarv2 ana "$label" -p min_track_length=160
done

# If pushd was executed, return to the original directory
if [ "$PARENT_DIR" == "Examples" ]; then
  popd
fi