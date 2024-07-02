# datathin

This is a Python version of data thinning, a method proposed by [Neufeld et al.](https://jmlr.org/papers/volume25/23-0446/23-0446.pdf). The code closely follows the original [R implementation](https://anna-neufeld.github.io/datathin/index.html) and uses a similar syntax. For example, to thin a vector of Poisson distributed `data`, simply call the `datathin` function, specifying the `family` and `epsilon` parameters:

```
from datathin.thinning import datathin

X = datathin(data, family="poisson", epsilon=[0.3, 0.7])
```

Here, `X` is a three-dimensional `numpy` array, and the folds of data are stored in the third dimension. I.e., 

```
X_train = X[:,:,0]
X_test = X[:,:,1]
```

Note that 1) `data == X_train + X_test`, and 2) the correlation between `X_train` and `X_test` is approximately zero. For more examples of how to thin data following other distributions, see `test_helpers.py`.

This Python project is very new, and contributions are very much welcome! If you find any bugs, feel free to open a GitHub issue or a pull request. Other helpful contributions include:

- Documentation (we currently have none)
- Optimizations that make the `datathin` function run faster
- Additional tests, particularly for different input shapes


## Installation

`datathin` is not yet available on PyPI and must be installed directly from this repo. First, initialize and activate a conda environment.

```
conda create -n datathin-dev python=3.10
conda activate datathin-dev
```

In the same directory as the `pyproject.toml` file, you can install dependencies (`numpy` is the only one) by running:

```
cd $GIT_PROJECT_HOME
pip install .
```

OR to install `datathin` in editable mode for development purposes, run:

```
pip install -e .[dev]
```

The `[dev]` suffix installs additional dependencies for contributors, including `pytest`.


## Citation

If you found this package useful, please cite the original data thinning paper.

```
@article{JMLR:v25:23-0446,
  author  = {Anna Neufeld and Ameer Dharamshi and Lucy L. Gao and Daniela Witten},
  title   = {Data Thinning for Convolution-Closed Distributions},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {57},
  pages   = {1--35},
  url     = {http://jmlr.org/papers/v25/23-0446.html}
}
```
