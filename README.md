# EHT 2018 M87* Visibility Domain Analysis

**Authors:** The Event Horizon Telescope Collaboration et al. <br>
**Primary Reference:** The Event Horizon Telescope Collaboration, et al. 2024, A&A, 681, A79  <br>
**Data Product Code:** 2024-D01-01  <br>

**Brief Description:**
This is a script that produces a series of posterior samples for the various geometric models used in the analysis of the 2018 M87 data. 
The primary workhorse is a Bayesian modeling package for radio astronomy applications[^PT].

The data for this analysis is available on the [EHT website](https://eventhorizontelescope.org/) (data release ID: 2024-D01-01).

## Additional References:
 - **EHT Collaboration Data Portal Website**:
   https://eventhorizontelescope.org/for-astronomers/data
 - **2018 M87 Paper 1**: The Event Horizon Telescope Collaboration, et al. 2024, A&A, 681, A79
 - **Comrade.jl**: https://github.com/ptiede/Comrade.jl 
 - **Dynesty**: https://dynesty.readthedocs.io/en/stable/

## Pre-requisites

### Julia < 1.9
```
juliaup add 1.9.4
juliaup default 1.9.4
```

## Running 

The script that generates the data for this analysis is located in `src/sample_models.jl`.
This script lists all the models used in the 2018 M87* analysis, and samples the posterior for each model using `dynesty`[^D].

1. Edit the `sample_models.jl` file in the `src/` folder to point to your directory of data files.

2. Include the number of child processes you want to use for parallel computation.

## References
[^PT]:Tiede, P. (2022). Comrade: Composable Modeling of Radio Emission. Journal of Open Source Software, 7(76), 4457. doi:10.21105/joss.04457

[^D]: Speagle, J. S. (2020). DYNESTY: a dynamic nested sampling package for estimating Bayesian posteriors and evidences. \mnras, 493(3), 3132–3158. doi:10.1093/mnras/staa278