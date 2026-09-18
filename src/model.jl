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

