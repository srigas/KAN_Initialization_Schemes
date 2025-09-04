# Introduction

This repository contains the code and experimental results for the paper "[Initialization Schemes for Kolmogorov–Arnold Networks: An Empirical Study](https://arxiv.org/abs/2509.03417)".


# Getting Started

After cloning the repository,

```bash
git clone https://github.com/srigas/KAN_Initialization_Schemes.git kan_init
cd kan_init
```

create a Python virtual environment, activate it and install all dependencies:

```bash
python3 -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
pip3 install -r requirements.txt
```

Then launch JupyterLab:

```bash
jupyter lab
```

Open the notebooks in order (`1.*.ipynb` → `8.*.ipynb`) to reproduce all experiments, including the grid search, PDE benchmarks, NTK analysis, and final plots presented in the paper.


# Citation

If the code and/or results presented here helped you for your own work, please cite our work as:

```
@misc{kaninit, 
	title = {Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study}, 
	author = {Spyros Rigas and Verma Dhruv and Georgios Alexandridis and Yixuan Wang}, 
	year = {2025}, 
	eprint = {2509.03417}, 
	archivePrefix = {arXiv}, 
	primaryClass = {cs.LG}, 
	url = {https://arxiv.org/abs/2509.03417}
}
```
