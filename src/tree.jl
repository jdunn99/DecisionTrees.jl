# Node structure for tree
abstract type Node end
struct Leaf{T}<: Node
	prediction::T
end

# Defines a split node with feature, threshold, left, right
struct Root{T} <: Node
	feature::Symbol
	threshold::T
	left::Node
	right::Node
end

# function prediction(::ClassificationCriterion, target_values::AbstractVector)nd
prediction(::ClassificationCriterion, target_values::AbstractVector) = argmax(unique_classes(target_values))
prediction(::RegressionCriterion, target_values::AbstractVector) = round(sum(target_values) / length(target_values), digits=4)

# Recursively predict a row in a DataFrame (aka a single observation)
function predict_row(node::Leaf, _)
	return node.prediction
end

# Move until we heat a leaf
# For now row remains untyped because of the NamedTuple. Could use union but not worth it if I may change it later.
# TODO: Define a type
function predict_row(node::Root, row)
	if row[node.feature] <= node.threshold
		return predict_row(node.left, row)
	else
		return predict_row(node.right, row)
	end
end

# Predict on an entire DataFrame
function predict(tree::Node, data::AbstractDataFrame)
	return [predict_row(tree, row) for row in eachrow(data)]
end

# May switch NamedTuple to a different data structure.
predict(tree::Node, observation::NamedTuple) = predict_row(tree, observation)

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
	# @show best_split(data, target_values, features, criterion)

	# TODO: Make a dispatched prediction function to handle criterion.
	# This only works with classification right now.
	length(unique(target_values)) == 1 && return Leaf(target_values[1])


	split = best_split(data, target_values, features, criterion)

	split.gain == -Inf && return Leaf(argmax(classes))

	# Split the tree based on the best threshold value
	best_feature = data[!, split.feature]
	threshold_split = best_feature .<= split.threshold

	left_split = @view data[threshold_split, :]
	right_split = @view data[.!threshold_split, :]

	# Recurse on subtrees
	left = fit_tree(left_split, target, features, criterion)
	right = fit_tree(right_split, target, features, criterion)

	return Root(split.feature, split.threshold, left, right)
end