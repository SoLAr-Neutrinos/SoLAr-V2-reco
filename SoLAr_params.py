import numpy as np

sipm_map_file = "sipm_sn_ch_to_xy.json"

match_dict = {}

# Load options
reload_files = True
rematch_events = False

# Save options
overwrite_metrics = True
save_figures = True

# Plotting options
flip_x = True
individual_plots = np.arange(1, 101, 1)
show_figures = False
label_font_size = 16
tick_font_size = 16
title_font_size = 18

# Events to process
event_list = None

# Noisy Pixels
channel_disable_list = [(7, (1, 2))]  # (chip, channel)

# Light variable to consider
light_variable = "integral"

# Units for plot labels
q_unit = "e"  # After applying charge_gain
xy_unit = "mm"
z_unit = "mm"
dh_unit = "?" if z_unit != xy_unit else xy_unit
time_unit = "ns"
light_unit = "p.e." if light_variable == "integral" else "p.e./time bin"

# Conversion factors
charge_gain = 245  # mV to e
detector_z = 300
detector_x = 128
detector_y = 160
sipm_size = 6
pixel_size = 3
pixel_pitch = 4
quadrant_size = 32  # One SiPM + LArPix cell
first_chip = (2, 1) if detector_y == 160 else (1, 1)

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

# Light variable to consider
light_variable = "integral"

# Filters for post processing if not using filter parameters file
score_cutoff = -1.0
max_score = 1.0
min_track_length = 160
max_track_length = np.inf
max_tracks = 1
max_light = np.inf
min_light = 0
max_z = np.inf

# Other
non_track_keys = 5
