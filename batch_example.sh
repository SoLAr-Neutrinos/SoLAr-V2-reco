# # Full analysis
# for light_file in /eos/experiment/dune/solar/data/SoLAr_v2/Light/root/46v_12db_th950_deco/*
# do
#   # Print the current file name
#   echo Light file: "$light_file"

#   # Extract the timestamp after 'hit_' and before '.root' using sed to get the output folder
#   label=$(echo $light_file | sed -n 's/.*_\([0-9]\{8\}_[0-9]\{6\}\)\.data\.root/\1/p')
  
#   # Transform the label to the charge file timestamp format
#   formatted_date=$(echo $label | sed 's/\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)_\([0-9]\{2\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)/\1_\2_\3_\4_\5/')

#   charge_file="/eos/experiment/dune/solar/data/SoLAr_v2/Charge/cosmics/root/evd_self_trigger-packets-${formatted_date}_CEST_validated.root"
#   echo Charge file: "$charge_file"
  
#   # Run the reconstruction script again without simulating dead areas
#   ./reconstruction.py -c "$charge_file" -l "$light_file" 

#   ./display_events.py "$label" -s -n
  
#   # Run the analysis script again on the output folder
#   ./analysis.py "$label"
#   ./analysis.py "$label" -p min_track_length=160
# done

# # Reconstruction after root files have been processed
# for folder in 202307*
# do
#   # Print the current file name
#   echo Folder: "$folder"
  
#   # Run the reconstruction script again without simulating dead areas
#   ./reconstruction.py -f "$folder"

#   ./display_events.py "$folder" -s -n
  
#   # Run the analysis script again on the output folder
#   ./analysis.py "$folder"
#   ./analysis.py "$folder" -p min_track_length=160
# done

# Only event displays and analysis after
for folder in 202307*
do
  # Print the current file name
  echo Folder: "$folder"

  ./display_events.py "$folder" -s -n
  
  # Run the analysis script again on the output folder
  ./analysis.py "$folder"
  ./analysis.py "$folder" -p min_track_length=160
done

# Run analysis on combined metrics
./analysis.py
./analysis.py -p min_track_length=160