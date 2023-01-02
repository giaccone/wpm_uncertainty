# wpm_uncertainty
Uncertainty Quantification of the Weighted Peak Method through Polynomial Chaos Expansion

This repository includes all Python codes related to the paper.

L. Giaccone, *"Uncertainty quantification in the assessment of human exposure to pulsed or multi-frequency fields"*.

This paper is currently under review for possible publication on an international journal.

## 1. Requirements

### 1.1 Python version
All codes have been tested with Python 3.9 and 3.10.

### 1.2 Python modules (available on conda and/or Pypi)

In order to run the Python codes the following modules are required:

* numpy (available on conda and Pypi)
* scipy (available on conda and Pypi)
* matplotlib (available on conda and Pypi)
* progress (available on conda and Pypi)
* joblib (available on conda and Pypi)
* pce (available Pypi, see here for more details: [https://github.com/giaccone/pce](https://github.com/giaccone/pce))

### 1.3 Other Python (non-packaged) projects
Finally, other Python codes are used in this repository that are not available on Pypy. It is necessary to download them and setup the interpreter in order to make them accessible in the path. This is the list of the required repositories:

* [https://github.com/giaccone/fourier](https://github.com/giaccone/fourier)
* [https://github.com/giaccone/exposure](https://github.com/giaccone/exposure)

## 2. Usage

First of all:

* clone the repository (`git clone https://github.com/giaccone/wpm_uncertainty`)
* read the file `currents.txt` (included in the `currents` folder). It includes instructions about how to download the measured data.

Afterward:

* activate a Python environment with the requirements described in section 1

Finally:

* each script on the repository can be executed