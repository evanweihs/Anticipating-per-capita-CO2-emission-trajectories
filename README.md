# Anticipating-per-capita-CO2-emission-trajectories
This project implements a fully reproducible machine-learning pipeline to analyse and predict country-level trajectories of per capita CO₂ emissions. The workflow combines data preparation, dimensionality reduction and supervised models, with a strong emphasis on modularity, temporal validation and reproducibility.

Research Question
Can machine learning models be used to group countries according to their per capita CO₂ emissions profiles and predict future emissions trends based on economic data and energy structure?
Data

    Unit of observation: Country-Year
    Time span: 2000-2023
    Sources:
        World Bank, World Development Indicators: GDP per capita, population
        Our World in Data: CO₂ emissions per capita; energy consumption per capita by source (coal, oil, gas, nuclear, hydro, wind, solar, other renewables)
        Panel construction: Countries are harmonised using consistent ISO-3 identifiers Regional aggregates and non-country entities are excluded

The final dataset is a balanced country–year panel after cleaning and alignment across sources (see data preparation notebooks)
Outcome Variable

The outcome variable is per capita CO₂ emissions at the country–year level.

For the predictive task, the target variable yc,t corresponds to the level of CO₂ emissions per capita in year t. Models are trained in a temporal setting, using economic and energy related covariates observed at t−1, in order to predict future emission levels while preventing information leakage.

Feature Set

The feature set is composed of economic and energy-related variables observed at the country–year level.

  Core covariates include:

  GDP per capita

  
  
  Population

Energy consumption per capita by source, including coal, oil, gas, nuclear, hydro, wind, solar, and other renewables

All features are harmonised across data sources using ISO-3 country codes. Variables are expressed in per capita terms when relevant to ensure cross-country comparability.

To introduce temporal dynamics, lagged versions of the explanatory variables are constructed and used as inputs for the supervised models. No contemporaneous or future information on emissions is included in the feature set to prevent target leakage.
	​

