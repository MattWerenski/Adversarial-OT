# Adversarial Learning for Multiclass Optimal Transport

Github Repo for Topic 9. Inspired by the work in [this paper.](https://arxiv.org/pdf/2204.12676.pdf)

## Setup

Setting up the environment requires [Conda.](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) Install this first if you don't already have it. Then run the following sequence of commands to get the standard packages (more may be added later).

```
# create a fresh environment
conda create --name [envname] python=3.7

conda activate [envname]

# install a bunch of packages from conda forge
conda install -c conda-forge numpy scipy pot matplotlib tqdm jupyterlab cvxopt
pip install miniball
```
