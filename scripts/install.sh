source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
python -m venv venv
source $PWD/venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name venv