"""
	Model{T}

The result of fitting and cross-validating a decision tree via [`fit`](@ref).
Bundles the tree itself with everything needed to build a complexity parameter table.

# Fields
- `tree`: The unpruned base tree
- `criterion`: The [`Criterion`](@ref) used to build the tree
- `target`: The column being predicted
- `features`: The columns used in splitting
- `minsplit`, `maxdepth`: Stopping parameters used in building the tree
- `root_error`: Unpruned root error used to normalize `xerror`
- `n`: The number of training rows
- `events`: The [`PruneEvent`](@ref) result from [`generate_alphas`](@ref)
- `thresholds`, `xerror`, `xstd`: Per level cross validation results from [`cross_validate`](@ref)
"""
struct Model{T}
	tree::TNode{T}
	criterion::Criterion
	target::Symbol
	features::Vector{Symbol}
	minsplit::Int
	maxdepth::Int
	root_error::Float64
	n::Int
	events::Vector{PruneEvent{T}}
	thresholds::Vector{Float64}
	xerror::Vector{Float64}
	xstd::Vector{Float64}
end

"""
	fit(data, target, features, criterion, minsplit = 10, maxdepth = 5, k = 10)

Fits a decision tree and cross validate using cost-complexity pruning.
Builds an unpruned tree, computes the alpha sequence, and runs K-fold CV.

Returns a [`Model`](@ref) holding all results ready for analysis.

# Arguments
- `data`: Training data to build model
- `target`: Column to predict
- `target`: The column being predicted
- `features`: The columns used in splitting
- `minsplit`, `maxdepth`: Stopping parameters used in building the tree
- `k`: Number of cross validation folds (default `10`)

# Examples
```jldoctest
julia> using DecisionTrees, DataFrames

julia> data = DataFrame(x = collect(1.0:20.0), y = [fill(0.0, 10); fill(1.0, 10)]);

julia> model = fit(data, :y, [:x], MSECriterion(); minsplit=2, maxdepth=3, k=5);

julia> model.n
20
```
"""
function fit(
	data::AbstractDataFrame,
	target::Symbol,
	features::Vector{Symbol},
	criterion::Criterion,
	minsplit::Int = 10,
	maxdepth::Int = 5,
	k::Int = 10
)
	tree = fit_tree(data, target, features, criterion, minsplit, maxdepth)
	root_error = tree.error
	n = nrow(data)
	events = generate_alphas(tree)
	cv = cross_validate(data, target, features, criterion, events, root_error, k, minsplit, maxdepth)

	return Model(tree, criterion, target, features, minsplit, maxdepth, root_error, n, events, cv.thresholds, cv.xerror, cv.xstd)
end

# Section still a work in progress. Not 100% sure about how to format printing / plotting.
# Basically copy rpart or something else?

function print_cp(model::Model)
	println("Root node error: ", round(model.root_error, digits=5), "/", model.n,
	        " = ", round(model.root_error / model.n, digits=5))
	println()
	println("n= ", model.n)
	println()

	rows = reverse(model.events)
	println(rpad("", 3), rpad("CP", 10), rpad("nsplit", 8), rpad("rel error", 11),
	        rpad("xerror", 9), "xstd")
	for (i, e) in enumerate(rows)
		idx = length(model.events) - i + 1  
		cp = round(e.alpha / model.root_error, digits=6)
		println(rpad(i, 3), rpad(cp, 10), rpad(e.number_splits, 8),
		        rpad(round(e.rel_error, digits=5), 11),
		        rpad(round(model.xerror[idx], digits=5), 9),
		        round(model.xstd[idx], digits=6))
	end
end

function plot_cp(model::Model)
	# We need events, root_error, xerr, xstd

	cps = [event.alpha / model.root_error for event in model.events]
	nsplits = [event.number_splits for event in model.events]

	# Filter 0.0 cp values for log plotting
	min_cp = minimum(cp for cp in cps if cp > 0)
	cps = [cp > 0 ? cp : min_cp * 0.5 for cp in cps]

	best_index = argmin(model.xerror)

	# One standard error rule
	se_threshold = model.xerror[best_index] + model.xstd[best_index]

	plt = plot(
		cps, model.xerror,
		yerror = model.xstd,
		xflip = true,
		xscale = :log10,
		marker = :circle,
		linecolor = :black,
		markercolor = :black,
		label = "",
		xlabel = "cp",
		ylabel = "X-val relative error",
		title = "",
		legend = false
	)

	return plt
end