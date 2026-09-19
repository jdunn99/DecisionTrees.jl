# splitting.jl - Core recursive tree construction

"""
	best_split(data, target_values, features, criterion)

Find the feature and threshold that maximizes gain for a split of `target_values`.
Searches over every canditate threshold for every feature in `features`.

# Returns
`(gain, threshold, feature)`. `gain == -Inf` means not valid split was found.
"""
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
			threshold = unique_features[i]

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

"""
	fit_tree(data, target, features, criterion, minsplit=10, maxdepth=5, currentdepth=0)

Builds an unpruned CART decision tree via recursive binary splitting.
Uses [`best_split`](@ref) to choose each split and `criterion` to score possible splits.

Returns the root [`TNode`](@ref).

# Arguments
- `data`: The training data to build the tree
- `target`: Column to predict
- `features`: Candidate split columns
- `criterion`: [`Criterion`](@ref) used in classification / regression.
- `minsplit`: Minimum node size to split (default `10`)
- `maxdepth`: Maximum recursion depth (default `5`)
- `currentdepth`: Internal recursion counter. NOTE: Leave at default when calling `fit_tree` directly.

# Examples
```jldoctest
julia> using DecisionTrees, DataFrames

julia> data = DataFrame(x = [1.0, 2.0, 3.0, 4.0], y = [0.0, 0.0, 1.0, 1.0]);

julia> tree = fit_tree(data, :y, [:x], MSECriterion());

julia> tree.is_leaf
false
```
"""
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
	pred = prediction(criterion, target_values)
	current_error = tree_error(criterion, target_values, pred)

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