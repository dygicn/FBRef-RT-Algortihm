# FBRef-RT Algorithm

This repository contains the implementation of the FBRef-RT algorithm proposed for refining CART-based regression models near decision boundaries.

## Overview

FBRef-RT (Fuzzy Boundary Refinement for Regression Trees) is a local post-hoc refinement framework. 
The method preserves the interpretable structure of the original CART model while selectively improving unstable routing decisions.

The proposed framework:

- represents uncertainty by Gaussian fuzzy numbers (GFNs),
- uses GFN-based arithmetic operations throughout the refinement process,
- evaluates Apriori-based support and confidence measures in the GFN domain,
- uses Kullback-Leibler divergence as a proximity criterion near split boundaries,
- accepts local routing refinements only when validation performance improves.

The framework is studied for:

- CART
- Bagging
- Random Forest

## Repository structure

This repository will contain the scripts and outputs related to:

- CART+FBRef-RT
- Bagging+FBRef-RT
- Random Forest+FBRef-RT
- hyperparameter tuning
- runtime analysis
- performance comparison plots

## Requirements

The implementation is written in R.

Required packages include:

- `rpart`
- `openxlsx`
- `readr`
- `ggplot2`

Additional packages may be required depending on the script.

## Input data

The experiments are based on train and test datasets stored in CSV format. Train and test files should have matching names.

## Outputs

The scripts produce:

- best hyperparameter configurations,
- dataset-level performance summaries,
- runtime summaries,
- random-search outputs,
- result plots.
