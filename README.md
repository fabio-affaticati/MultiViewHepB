## Table of Contents
- [Multi-view learning to unravel the different levels underlying hepatitis B vaccine response](#pydeseq2)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
    - [1 - Download the repository](#1---download-the-repository)
    - [2 - Create a virtual environment](#2---create-a-virtual-environment)
    - [Requirements](#requirements)
  - [Data](#data)
  - [Citing this work](#citing-this-work)
  - [References](#references)


## Overview

The high complexity of biological systems arises from the large number of spatially and functionally overlapping interconnected components constituting them. The immune system, which is built of reticular components working to ensure host survival from microbial threats, presents itself as particularly intricate. A vaccine response is likely governed by levels that, when considered separately, may only partially explain the mechanisms at play. Multi-view modelling can aid in gaining actionable insights on response markers shared across populations, capture the immune system diversity, and disentangle confounders. Hepatitis B virus (HBV) vaccination responsiveness acts as a feasibility study for such an approach. Multi-view dimensionality reduction complements the clinical seroconversion and all single-modalities and could identify what features underpin HBV vaccine response. This methodology could be extended to other vaccination trials to identify key features regulating responsiveness.


## Installation


### 1 - Download the repository

`git clone https://github.com/.git`


### 2 - Create a virtual environment

`cd` to the root of the repo and use the venv package to create a virtual environment:
run `python3 -m venv <env>` to create the `<env>` environment and then activate it with `source env/bin/activate`.

```
python3 -m venv <env>
source env/bin/activate
```

Or use a conda environment:
run `conda env create -n <env>` to create the `<env>` environment and then activate it with `conda activate <env>`.

```
conda create -n <env>
conda activate <env>
```

Finally, install required packages with:

```
pip install -r requirements.txt
```


### Requirements

The code was tested with Python versions 3.9.6 and 3.11.2.
Due to the impossibility of solving a conflict of packages, the py_deseq.py file was added directly to the project.
Credit to [pydeseq2](https://github.com/owkin/PyDESeq2).

The list of package version requirements below is available in `requirements.txt` and it is used to reproduce the original virtual environment.

```
- Bio==1.5.9
- imbalanced_learn==0.10.1
- imblearn==0.0
- ipykernel==6.22.0
- joblib==1.2.0
- kaleido==0.2.1
- matplotlib==3.3.4
- mvlearn==0.5.0
- natsort==8.3.1
- nbformat==5.8.0
- numpy==1.24.2
- pandas==1.5.3
- plotly==5.13.1
- rpy2==3.5.10
- scikit_learn==1.2.2
- scipy==1.10.1
- seaborn==0.11.2
- statsmodels==0.13.5
- openpyxl==3.1.2
```


## Data

The article based on the code and the analysis here conducted is currently in [preprint](https://www.biorxiv.org/content/10.1101/2023.02.23.529670v1). 
This study relied on data first introduced by [1] and [2]. The datasets can be found under the `data` directory.


## Citing this work

```
@article {Affaticati2023.02.23.529670,
  title = {Multi-view learning to unravel the different levels underlying hepatitis B vaccine response},
	author = {Fabio Affaticati and Esther Bartholomeus and Kerry Mullan and Pierre Van Damme and Philippe Beutels and Benson Ogunjimi and Kris Laukens and Pieter Meysman},
	year = {2023},
	doi = {10.1101/2023.02.23.529670},
  journal={bioRxiv},
}
```

## References

[1]	E. Bartholomeus et al., ‘Transcriptome profiling in blood before and after hepatitis B vaccination shows significant differences in gene expression between responders and non-responders’,     Vaccine, vol. 36, no. 42, pp. 6282–6289, Oct. 2018, doi: 10.1016/j.vaccine.2018.09.001.
[2]	G. Elias et al., ‘Preexisting memory CD4 T cells in naïve individuals confer robust immunity upon hepatitis B vaccination’, eLife, vol. 11, p. e68388, Jan. 2022, doi: 10.7554/eLife.68388.
