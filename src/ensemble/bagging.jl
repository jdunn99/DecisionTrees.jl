function bootstrap(data::AbstractDataFrame, 
				   target::Symbol, 
				   features::Vector{Symbol}, 
				   criterion::Criterion, 
				   num_trees::Int, 
				   minsplit::Int=10, 
				   maxdepth::Int=5,
				   mtry::Union{Int, Nothing}=nothing
)
	n = nrow(data)

	T = eltype(data[!, target])
	trees = Vector{TNode{T}}(undef, num_trees)   # not Vector{TNode}

	for i in 1:num_trees
		samples = rand(1:n, n)	
		bootstrap_data = @view data[samples, :]
		trees[i] = fit_tree(bootstrap_data, target, features, criterion, minsplit, maxdepth, mtry)
	end

	return trees
end

function bag(data::AbstractDataFrame, target::Symbol, features::Vector{Symbol}, criterion::Criterion, num_trees::Int, minsplit::Int=10, maxdepth::Int=5)
	trees = bootstrap(data, target, features, criterion, num_trees, minsplit, maxdepth)
	return BaggingTreeModel(trees, criterion, target, features, minsplit, maxdepth)
end

# Take an array of predictions and aggregate them
function aggregate_predictions(::RegressionCriterion, predictions::Vector{<:AbstractVector})
	return [sum(row) / length(row) for row in zip(predictions...)]
end

function aggregate_predictions(::ClassificationCriterion, predictions::Vector{<:AbstractVector})
	return [argmax(unique_classes(collect(row))) for row in zip(predictions...)]
end