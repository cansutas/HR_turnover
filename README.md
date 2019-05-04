# HR_turnover
Predicting turnover of employees using Random Forest algorithm. 

The datasets used here are not public datasets, therefore are not uploaded here. 
The data:
* main: the data with basic demographics
* attend: working hours entered
* absencence: vacation hours entered
* travel: travel hours entered

## Initial File Structure

```
├── data
│   ├── interim_[desc]       <- Interim files - give these folders whatever name makes sense.
│   ├── processed            <- The final, canonical data sets for modeling.
│   ├── raw                  <- The original, immutable data dump.
│   └── temp                 <- Temporary files.
│
│
├── notebooks                <- Notebooks for analysis and testing
│   ├── eda                  <- Notebooks for EDA
│   │   └── example.ipynb    <- Example python notebook
│   ├── features             <- Notebooks for generating and analysing features (1 per feature)
│   ├── modelling            <- Notebooks for modelling
│   └── preprocessing        <- Notebooks for Preprocessing 
│
├── reporting                <- Solutions for reporting of results
│   ├── webapp               <- Flask based template for displaying content including text and graphs
│   └── README.md            <- Information on usage and setup of the webapp sample and more
│
├── src                      <- Code for use in this project.
│   └── examplepackage       <- Example python package - place shared code in such a package
│       ├── __init__.py      <- Python package initialisation
│       ├── examplemodule.py <- Example module with functions and naming / commenting best practices
│       ├── features.py      <- Feature engineering functionality
