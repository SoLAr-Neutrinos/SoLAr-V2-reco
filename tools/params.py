# This file contains the default parameters used by the methods.py file and the notebooks.
# Change parameters on your own script by calling params.PARAMETER
import os

import numpy as np

sipm_map_file = os.path.abspath(
    os.path.join(__file__, os.pardir, "sipm_sn_ch_to_xy.json")
)

# Load options
reload_files = True
simulate_dead_area = False
rematch_events = False

# Save options
work_path = os.getcwd()
output_folder = "draft"
overwrite_metrics = True
save_figures = True

# Plotting options
flip_x = True
individual_plots = np.arange(1, 101, 1)
show_figures = False
label_font_size = 16
tick_font_size = 16
title_font_size = 18
style = "default"  # "dark_background"

# Events to process
event_list = None

# Noisy Pixels
channel_disable_list = [(7, (1, 2))]  # (channel, (x coord, y coord))

# Light variable to consider
light_variable = "integral"

# Units for plot labels
q_unit = "e"  # After applying charge_gain
xy_unit = "mm"
z_unit = "mm"
time_unit = "ns"
dh_unit = "mm"
light_unit = "p.e."


# Conversion factors
charge_gain = 245  # mV to e
detector_z = 300
detector_x = 128
detector_y = 160
sipm_size = 6
pixel_size = 3
pixel_pitch = 4
quadrant_size = 32  # One SiPM + LArPix cell
first_chip = (2, 1)


# DBSCAN parameters for charge clustering
min_samples = 2
xy_epsilon = 8  # 8 ideal
z_epsilon = 8  # 8 ideal

# RANSAC parameters for line fitting
ransac_residual_threshold = 6  # 6 ideal for charge, 35 ideal for light
ransac_max_trials = 1000
ransac_min_samples = 2  # 2 ideal for charge, 3 ideal for light

# Force parameters for cylinder
force_dh = None
force_dr = None

# Filters for post processing if not using filter parameters file
filter_label = None
min_score = -1.0
max_score = 1.0
min_track_length = 32
max_track_length = np.inf
max_tracks = 1
max_light = np.inf
min_light = 0
max_z = np.inf
