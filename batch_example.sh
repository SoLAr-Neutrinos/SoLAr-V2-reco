# Loop through all files matching the pattern in the specified directory
for file in ../../data/SoLAr_v2/Light/root/46v_12db_th950_deco/*
do
  # Print the current file name
  echo $file
  
  # Extract the numbers after 'hit_' and before '.root' using sed to get the output folder
  label=$(echo $file | sed -n 's/.*_\([0-9]\{8\}_[0-9]\{6\}\)\.data\.root/\1/p')
  
  # Run the reconstruction script again without simulating dead areas
  ./reconstruction.py $file -p force_dh=30
  
  # Run the analysis script again on the output folder
  ./analysis.py $label -s
done