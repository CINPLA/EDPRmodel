[![Build Status](https://app.travis-ci.com/CINPLA/EDPRmodel.svg?branch=master)](https://app.travis-ci.com/CINPLA/EDPRmodel)

# EDPRmodel

EDPRmodel is an implementation of the KNP continuity equations for a
one-dimensional system containing two plus two compartments.
The model is presented in Sætra et al., *PLoS Computational Biology*, 16(4), e1007661 (2020): [An electrodiffusive, ion conserving Pinsky-Rinzel model with homeostatic mechanisms](https://doi.org/10.1371/journal.pcbi.1007661
).

## Installation 

Clone or download the repo, navigate to the top directory of the repo and enter the following
command in the terminal: 
```bash
python setup.py install
```
The code was developed for Python 3.6.

## Run simulations

The simulations folder includes example code showing how to run simulations. 
To reproduce the results presented in Sætra et al. 2020, see 
[https://github.com/CINPLA/EDPRmodel_analysis](https://github.com/CINPLA/EDPRmodel_analysis).
