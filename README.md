Welcome to the SoLAr-V2-reco repository! This repository contains scripts, tools, and resources for the SoLAr V2 data reconstruction.

## Installation

To get started, follow these steps to install the required dependencies:

1. Clone the repository: `git clone https://github.com/SoLAr-Neutrinos/SoLAr-V2-reco.git`
2. Navigate to the `SoLAr-V2-reco` directory: `cd SoLAr-V2-reco/`
3. Install on LXPLUS: `sh install_lxplus.sh` 
    or
    Install only the dependencies: `pip install -r requirements.txt`

## Features

The SoLAr-V2-reco repository provides the following features:

- **Bash Script**: An example bash script (`batch_example.sh`) is included, demonstrating the analysis process.
- **Jupyter Notebooks**: Interactive step-by-step Jupyter notebooks for the reconstruction and analysis are available for a user-friendly experience.
- **Python Wrappers**: Python wrappers for modular reconstruction, analysis, and event display are also provided.

## Usage

To incorporate the `tools` package into your own analysis, simply import it using the following code snippet:

```python
from solarv2 import *
```

For Monte Carlo analysis, a dedicated package named `solarv2-mc` is available. The base package is intended for data analysis.

To run solarv2 from the command line, you can call:

1. For the reconstruction workflow
```python
solarv2 reco
```
1. For producing event displays
```python
solarv2 display
```

1. For the analysis workflow
```python
solarv2 ana
```

Use the `--help` flag to view the input parameters.

A dictionary of custom parameters can be passed as input. These are:

**Loading and saving options:**
- reload_files: whether to reload ROOT files instead of using saved bz2 dataframes.
- rematch_events: whether to recreate the charge/light event match dictionary.
- output_folder: name of the output folder. Automatically defined by the wrappers. Use only as an override in your own code.
- overwrite_metrics: whether to overwrite the metrics file if it already exists.
- save_figures: whether to save plots to disk.

**Plotting:**
- individual_plots: list of events for which to make individual plots
- show_figures: whether to show each individual plot instead of just saving to disk.
- label_font_size: plot label size.
- tick_font_size: plot tick label size.
- title_font_size: plot title font size.

**Reconstruction exclusions:**
- event_list: list of events to load from ROOT files. If null, loads all events.
- channel_disable_list": list of channels to disable. Format is (channel_number, (channel x coordinate, channel y coordinate)) where teh coordiantes are respective to the position in the chip's quadrant.

**Units and conversions:**
- light_variable: Which reconstructed light variable to use in the analysis. Options are integral or peak.
- q_unit: Charge unit label for plots.
- xy_unit: x and y coordinates unit label for plots.
- z_unit: z coordinate unit label for plots.
- time_unit: time unit label for plots.
- dh_unit: length unit label for dQ/dx plots.
- light_unit: light unit label for plots.
- charge_gain: ADC to electrons conversion factor.
  
**Detector geometry:**
- flip_x: whether to flip the x-axis. Needed for data files as the coordiante system was not defined right-handed.
- detector_z: detector z length.
- detector_x: detector x length.
- detector_y: detector y length.
- sipm_size: sipm size (6 mm).
- pixel_size: pixel size (3 mm).
- pixel_pitch: separation between pixels (4 mm)
- quadrant_size: size of a single chip/SiPM sector (32 mm)
- first_chip: internal parameter to determine the first chip in the grid when generatin dead areas map.
  
**DBSCAN:**
- min_samples: DBSCAN min_samples.
- xy_epsilon: DBSCAN epsilon for clustering in the xy plane.
- z_epsilon: DBSCAN epsilon for clustering along the z axis.
  
**RANSAC:**
- ransac_residual_threshold: RANSAC parameter to set residuals threhsold.
- ransac_max_trials: RANSAC parameter to set the maximum number of trials.
- ransac_min_samples: minimum number of hits needed to apply RANSAC fit.
  
**dQ/dx:**
- force_dh: value to override dh calculation in dQ/dx function.
- force_dr: value to override dr calculation in dQ/dx function.
  
**Data filtering:**
- min_score: minimum RANSAC score to allow in data selection.
- max_score: maximum RANSAC score to allow in data selection.
- min_track_length: minimum track length to allow in data selection.
- max_track_length: maximum track length to allow in data selection.
- max_tracks: maximum number of tracks per event to allow in data selection.
- max_light: maximum amount of light to allow in data selection.
- min_light: minimum amount of light to allow in data selection.
- max_z: maximum z-coorsinate of hits to allow in data selection.

**Internal parameters (only chane if you know what you are doing):**
- non_track_keys: internal parmeter for fitting function.
- sipm_map_file: location of the SiPM map JSON file.

## Contributing

If you would like to contribute to the SoLAr-V2-reco project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b my-feature`
3. Make your changes and commit them: `git commit -am 'Add new feature'`
4. Push your changes to your forked repository: `git push origin my-feature`
5. Open a pull request on the main repository.

Feel free to explore and contribute to the SoLAr project!
