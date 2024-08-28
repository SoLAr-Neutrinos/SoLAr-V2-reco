#!/bin/sh

# Loop through all files matching the pattern in the specified directory
# for file in  $SOLAREOS/montecarlo/singlecube/cosmic_v0/singlecube_cry_hit_*.root
for file in ../Monte\ Carlo/singlecube_cry_hit_*.root
do

echo "Processing" $file "with dead areas"
# Extract the numbers after 'hit_' and before '.root' using sed to get the output folder
folder=$(echo $file | sed -n 's/.*hit_\([0-9]*\)\.root/\1/p')

# Run the reconstruction script simulating dead areas and dh set to 30
./reconstruction.py "$file" -d -p file_label=$folder

# Make event displays
./display_events.py $folder -n -d -s

# Run the analysis script on the output folder
./analysis.py $folder -d #-s

echo "Processing" $file "without dead areas"

# Run the reconstruction script again without simulating dead areas
./reconstruction.py "$file" -p file_label=$folder

# Make event displays
./display_events.py $folder -n -s

# Run the analysis script again on the output folder
./analysis.py $folder #-s

done
