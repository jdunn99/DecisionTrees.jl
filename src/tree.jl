""" Structures """ 

# Node structure for tree

# Works like a Tree class in C
mutable struct TNode{T} 
	is_leaf::Bool
	prediction::T
	error::Float64 # Used for pruning
	feature::Symbol
	threshold::Float64
	left::Union{TNode{T}, Nothing}
	right::Union{TNode{T}, Nothing}
	parent::Union{TNode{T}, Nothing}
	subtree_error::Float64
	number_leaves::Int

	# Leaf
	TNode(pred::T, err::Float64) where {T} = new{T}(true, pred, 
			err, :none, 0.0, nothing, nothing, nothing, err, 1)

	# Root
	function TNode(pred::T,
	 	 err::Float64, 
	 	 feat::Symbol, 
	 	 thres::Float64, 
	 	 l::TNode{T}, 
	 	 r::TNode{T}
	 ) where {T}
		node = new{T}(false, pred, err, feat, thres, l, r, nothing, 
			l.subtree_error + r.subtree_error, l.number_leaves + r.number_leaves)
		l.parent = node
		r.parent = node
	end

end

""" Metrics """
tree_error(::ClassificationCriterion, actual::AbstractVector, predicted::AbstractVector) = Float64(sum(actual .!= predicted))
tree_error(::RegressionCriterion, actual::AbstractVector, predicted::AbstractVector) = sum((actual .- predicted).^2)

is_attached(node::TNode) = node.parent === nothing || node.parent.left === node || node.parent.right === node

function get_subtree(node::TNode)
	nodes = TNode[]
	_get_subtree!(nodes, node)
	return nodes
end

# Helper recurse function
function _get_subtree!(nodes::Vector{TNode}, node::TNode)
	if !node.is_leaf
		push!(nodes, node)
		_get_subtree!(nodes, node.left)
		_get_subtree!(nodes, node.right)
	end
	return nothing
end

""" Tree Prediction """ 
# function prediction(::ClassificationCriterion, target_values::AbstractVector)nd
prediction(::ClassificationCriterion, target_values::AbstractVector) = argmax(unique_classes(target_values))
prediction(::RegressionCriterion, target_values::AbstractVector) = round(sum(target_values) / length(target_values), digits=4)

function predict_row(node::TNode, row)
	node.is_leaf && return node.prediction

	if row[node.feature] <= node.threshold
		return predict_row(node.left, row)
	else
		return predict_row(node.right, row)
	end
end

# Predict on an entire DataFrame
function predict(tree::TNode, data::AbstractDataFrame)
	return [predict_row(tree, row) for row in eachrow(data)]
end

# May switch NamedTuple to a different data structure.
predict(tree::TNode, observation::NamedTuple) = predict_row(tree, observation)

""" Building Tree """ 

# Calculate the split that minimizes loss using the provided criterion
function best_split(
	data::AbstractDataFrame,
	target_values::AbstractVector,
	features::Vector{Symbol},
	criterion::Criterion
)
	n = length(target_values)
	
	parent_loss = calculate_loss(criterion, target_values)	

	best_gain = -Inf
	best_feature = :none
	best_threshold = nothing

	# Determine which feature provides the best split
	for feature in features
		feature_values = data[!, feature]
		unique_features = unique(feature_values)
		# Sort each feature value to find optimal threshold
		sort!(unique_features)

		for i in 1:(length(unique_features) - 1)
			threshold = feature_values[i]

			# Split into binary array based on threshold
			threshold_split = feature_values .<= threshold

			if !any(threshold_split) || all(threshold_split)
				continue
			end

			# Split the data based on the threshold
			left_split = @view target_values[threshold_split]
			right_split = @view target_values[.!threshold_split]

			nl = length(left_split)
			nr = n - nl

			weighted_loss = (nl / n) * calculate_loss(criterion, left_split) +
				   (nr / n) * calculate_loss(criterion, right_split)
			gain = parent_loss - weighted_loss

			if gain > best_gain
				best_gain = gain
				best_threshold = threshold
				best_feature = feature
			end
		end

	end

	return (gain=best_gain, threshold=best_threshold, feature=best_feature)
end

# Take in a DataFrame, target, features, and a loss criterion
# Build an unpruned tree using recursive binary splitting
# Will eventually default to building pruned trees with a default cp value unless provided
# Maybe in the future will try and support the R style syntax Target ~ f1 + f2 + ...
function fit_tree(
	data::AbstractDataFrame,
	target::Symbol,
	features::Vector{Symbol},
	criterion::Criterion,
	minsplit::Int = 10,
	maxdepth::Int = 5,
	currentdepth::Int = 0,
)
	target_values = data[!, target]
	classes = unique_classes(target_values)
	pred = prediction(criterion, target_values)
	current_error = tree_error(criterion, target_values, fill(pred, length(target_values)))

	length(unique(target_values)) == 1 && return TNode(pred, current_error)
	currentdepth >= maxdepth && return TNode(pred, current_error)
	length(target_values) < minsplit && return TNode(pred, current_error)

	split = best_split(data, target_values, features, criterion)
	split.gain == -Inf && return TNode(pred, current_error)

	# Split the tree based on the best threshold value
	best_feature = data[!, split.feature]
	threshold_split = best_feature .<= split.threshold

	left_split = @view data[threshold_split, :]
	right_split = @view data[.!threshold_split, :]

	# Recurse on subtrees
	left = fit_tree(left_split, target, features, criterion, minsplit, maxdepth, currentdepth + 1)
	right = fit_tree(right_split, target, features, criterion, minsplit, maxdepth, currentdepth + 1)

	return TNode(pred, current_error, split.feature, split.threshold, left, right)
end