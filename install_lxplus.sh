#!/usr/bin/env bash

# Source ROOT and LCG packages FIRST
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh

# Create venv with --system-site-packages to access ROOT
python -m venv --system-site-packages venv
source $PWD/venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Install ipykernel
python -m ipykernel install --user --name venv