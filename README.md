# DecisionTrees.jl

A Julia implementation of CART decision trees, with cost-complexity pruning and cross-validation.
Inspired by R's `rpart`. 
<!-- The underlying math and general tree building information come from *ISLR* -->

Still an early work in progress. Supports basic tree building features

- **Fitting** - recursive binary splitting on features for both regression (MSE) and classification (Gini impurity)
- **Pruning** - uses cost-complexity pruning to generate candiate alpha levels.
- **Cross-validation** - k-fold cross validate of alpha candidates, calculates `xerror` and `xstd` at each level.
- **Prediction** - predict with a `DataFrame` or a single observation
- **Basic Printing & Plotting** - Includes a CP table and plot function for plotting `cp` values against `xerror`

## Example Use

```julia
using DecisionTrees, DataFrames
 
data = DataFrame(
    age            = [22, 35, 58, 41, 29, 63, 47, 19, 52, 38],
    missed_payment = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    default        = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
)
 
model = fit(data, :default, [:age, :missed_payment], GiniCriterion())
 
print_cp(model)          
plot_cp(model)            # requires `using Plots`
 
predict(model.tree, data) # predictions on the training data
```

## Current Limitations

- No categorical data for splitting
- `best_split` has performance issues with large datasets
- Cross-validation folds are random splits, leading to poor results for imblanaced classes

## Roadmap

Rough outline of planned work. Various features and optimizations might be implemented outside of the roadmap.

- [] **Categorical features and best_split optimization** - Change `best_split` to handle non-numeric features. Also work on performance with `best_split`.
- [] **Bagging and Ensemble Structure** - Basic bootstrap trees with average/majority predictions. Requires a `aggregate` function shared by other ensemble methods.
- [] **Random Forest** - Bagging plus per-split sampling.
- [] **Boosting** - Requires reworking `fit_tree` 
- [] **Deployment**

<!-- [![Build Status](https://github.com/jdunn99/DecisionTrees.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jdunn99/DecisionTrees.jl/actions/workflows/CI.yml?query=branch%3Amaster) -->
