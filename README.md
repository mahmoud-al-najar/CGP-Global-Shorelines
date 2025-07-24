# Interpretable Machine Learning for Shoreline Forecasting
Authors: M. Al Najar, D. G. Wilson, R. Almar, 2025.

This repository hosts the scripts used in the above-mentioned publication. 

### Datasets
Due to file size limiatitions, the raw evolved models archive, processed dataset and all raw data used here are provided in a separate repository: https://doi.org/10.5281/zenodo.16407877

### Script files
The following scripts are included:

1. <code>evolve_global_models.py</code>:

2. <code>global_results.py</code>: reads the outputs from many evolutions, groups the populations, performs the result analysis and produces the plots used in the paper.

### Environment setup
This project was developed using Julia Version 1.6.1 and relies heavily on two other Julia repositories:
* [CartesianGeneticProgramming.jl](https://github.com/mahmoud-al-najar/CartesianGeneticProgramming.jl) for the implementation of CGP.
* [Cambrian.jl](https://github.com/mahmoud-al-najar/Cambrian.jl) as the base evolutionary computation framework.

To setup the environment, clone this repository in addition to [CartesianGeneticProgramming.jl](https://github.com/mahmoud-al-najar/CartesianGeneticProgramming.jl) and [Cambrian.jl](https://github.com/mahmoud-al-najar/Cambrian.jl).
The julia environment that will be used to run the scripts is included in [CartesianGeneticProgramming.jl](https://github.com/mahmoud-al-najar/CartesianGeneticProgramming.jl). Before installing any dependencies, it is important to activate this environment:

```
shell> cd path/to/CartesianGeneticProgramming.jl/
pkg> activate .
(CartesianGeneticProgramming) pkg>
```

Head to the [Cambrian.jl](https://github.com/mahmoud-al-najar/Cambrian.jl) directory and put it in development:
```
shell> cd path/to/Cambrian.jl/
(CartesianGeneticProgramming) pkg> dev .
```

The rest of the dependencies can then be installed by:
```
shell> cd path/to/CartesianGeneticProgramming.jl/
(CartesianGeneticProgramming) pkg> instantiate
(CartesianGeneticProgramming) pkg> precompile
```
