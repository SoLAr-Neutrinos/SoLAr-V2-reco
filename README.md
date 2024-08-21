Welcome to the SoLAr-V2-reco repository! This repository contains scripts, tools, and resources for the SoLAr V2 data reconstruction.

## Installation

To get started, follow these steps to install the required dependencies:

1. Clone the repository: `git clone https://github.com/dune/solar.git`
2. Navigate to the `SoLAr-V2-reco/tools` directory: `cd SoLAr-V2-reco/tools/`
3. Install on LXPLUS: `sh install.sh` 
    or
    Install only the dependencies: `pip install -r requirements.txt`

## Features

The SoLAr-V2-reco repository provides the following features:

- **Bash Script**: An example bash script (`batch_example.sh`) is included, demonstrating the analysis process.
- **Jupyter Notebooks**: Interactive step-by-step Jupyter notebooks for the reconstruction and analysis are available for a user-friendly experience.
- **Python Scripts**: Standalone python scripts for reconstruction, analysis, and event display are also provided.

## Usage

To incorporate the `tools` package into your own analysis, simply import it using the following code snippet:

```python
from tools import *
```

For Monte Carlo analysis, a dedicated branch named `mc` is available. The main branch is intended for data analysis.

## Contributing

If you would like to contribute to the SoLAr-V2-reco project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b my-feature`
3. Make your changes and commit them: `git commit -am 'Add new feature'`
4. Push your changes to your forked repository: `git push origin my-feature`
5. Open a pull request on the main repository.

Feel free to explore and contribute to the SoLAr project!
