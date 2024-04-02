![Python Version: 3.8.5](https://img.shields.io/badge/Python%20Version-3.8.5-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen)](https://opensource.org/licenses/MIT)

# Synthetic coevolution reveals adaptive mutational trajectories of neutralizing antibodies and SARS-CoV-2

Implementation of the paper [[Synthetic coevolution reveals adaptive mutational trajectories of neutralizing antibodies and SARS-CoV-2](https://doi.org/10.1101/2024.03.28.587189)], by Roy Ehling, Mason Minot, Max Overath, Daniel Sheward, Jiami Han, Beichen Gao, Joseph Taft, Margarita Pertseva, Cedric Weber, Lester Frei, Thomas Bikias, Ben Murrell, and Sai Reddy.

## Table of contents
1. [Environment](#environment)
2. [Reproducing Study Results](#reproducing-study-results)
3. [Citation](#citation)

## Environment 

#### Conda

```console
cd envs
conda env create -f syn_coev.yml
conda activate syn_coev
```
#### venv
1. `python -m venv syn_coev`
2. Windows: `syn_coev\Scripts\activate.bat`
   Unix / MacOS: `source syn_coev/bin/activate`
3. in `envs/`, run `pip install -r requirements.txt`
	
## Reproducing Study Results

 1. Preprocessing.
 2. Model training and evaluation.
 3. Plot results.

### Step 1 - Preprocessing
Note: Data will be made available following publication.
In `scripts/` run `preprocessing.sh`.

### Step 2 - Model Training and Evaluation
Note: analysis run with torch 2.1.2+cu121, but environment contains torch 2.1.2

In `scripts/` run `train.sh`.
This will populate the folder `results/` with .csv files in the appropriate format for plotting in Step 3.

### Setp 3 - Plot Results
In `scripts/` run `plot.sh`.

## Citation

```
@article {Ehling2024.03.28.587189,
	author = {Roy A. Ehling, Mason Minot, Max D. Overath, Daniel J. Sheward, Jiami Han, Beichen Gao, Joseph M. Taft, Margarita Pertseva, C{\'e}dric R. Weber, Lester Frei, Thomas Bikias, Ben Murrell, and Sai T. Reddy},
	title = {Synthetic coevolution reveals adaptive mutational trajectories of neutralizing antibodies and SARS-CoV-2},
	year = {2024},
	doi = {10.1101/2024.03.28.587189},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}
```