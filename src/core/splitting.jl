# splitting.jl - Core recursive tree construction

"""
	best_split(data, target_values, features, criterion)

Find the feature and threshold that maximizes gain for a split of `target_values`.
Searches over every canditate threshold for every feature in `features`.

# Returns
`(gain, threshold, feature)`. `gain == -Inf` means not valid split was found.
"""
# function best_split(
# 	data::AbstractDataFrame,
# 	target_values::AbstractVector,
# 	features::Vector{Symbol},
# 	criterion::Criterion
# )
# 	n = length(target_values)
	
# 	parent_loss = calculate_loss(criterion, target_values)	

# 	best_gain = -Inf
# 	best_feature = :none
# 	best_threshold = nothing

# 	# Determine which feature provides the best split
# 	for feature in features
# 		feature_values = data[!, feature]
# 		unique_features = unique(feature_values)
# 		# Sort each feature value to find optimal threshold
# 		sort!(unique_features)

# 		for i in 1:(length(unique_features) - 1)
# 			threshold = unique_features[i]

# 			# Split into binary array based on threshold
# 			threshold_split = feature_values .<= threshold

# 			if !any(threshold_split) || all(threshold_split)
# 				continue
# 			end

# 			# Split the data based on the threshold
# 			left_split = @view target_values[threshold_split]
# 			right_split = @view target_values[.!threshold_split]

# 			nl = length(left_split)
# 			nr = n - nl

# 			weighted_loss = (nl / n) * calculate_loss(criterion, left_split) +
# 				   (nr / n) * calculate_loss(criterion, right_split)
# 			gain = parent_loss - weighted_loss

# 			if gain > best_gain
# 				best_gain = gain
# 				best_threshold = threshold
# 				best_feature = feature
# 			end
# 		end

# 	end

# 	return (gain=best_gain, threshold=best_threshold, feature=best_feature)
# end

function numerical_split(
	feature_values::AbstractVector,
	target_values::AbstractVector,
	criterion::Criterion	
)
	n = length(target_values)

	# Track the indices that sorts feature_values and order the target the same
	sort_indices = sortperm(feature_values)
	sorted_feature = feature_values[sort_indices]
	sorted_target = target_values[sort_indices]

	parent_loss = calculate_total_loss(criterion, sorted_target)
	left_state = new_state(criterion, sorted_target)
	right_state = init_state(criterion, sorted_target)

	best_gain = -Inf
	best_threshold = nothing

	i = 1
	while i <= n - 1
		value = sorted_feature[i]
		j = i
		# Skips all tied rows
		while j <= n && sorted_feature[j] == value
			push!(left_state, sorted_target[j])
			pop!(right_state, sorted_target[j])
			j += 1
		end

		num_left = j - 1
		num_right = n - num_left

		if num_left > 0 && num_right > 0
			loss = (num_left / n) * calculate_loss(left_state) + (num_right / n) * calculate_loss(right_state)
			gain = parent_loss - loss

			if gain > best_gain
				best_gain = gain
				best_threshold = value
			end
		end
		i = j
	end

	return (gain=best_gain, threshold=ThresholdSplitRule(best_threshold))
end

# WARNING: THIS FUNCTION IS VERY EXPENSIVE. THERE WILL NEED TO BE A LOT OF OPTIMIZATION LATER ON.
function categorical_split(
	feature_values::AbstractVector,
	target_values::AbstractVector,
	criterion::Criterion
)
	categories = unique(feature_values)	
	n = length(categories)

	n < 2 && return (gain=-Inf, threshold=nothing)

	m = length(target_values)
	parent_loss = calculate_total_loss(criterion, target_values)

	best_gain = -Inf
	best_left_set = nothing

	# Split each category into 2 non-empty groups. Keep the one with best gain. 
	# Basically just create the powerset and iterate.
	rest = categories[2:end]
	for i in 1:(2^(n-1) - 1)
		left_mask = Bool.(digits(i, base=2, pad=n-1))
		left_set = Set(rest[left_mask])

		left_els = feature_values .∈ Ref(left_set)
		num_left = count(left_els)
		num_right = m - num_left

		(num_left == 0 || num_right == 0) && continue

		loss = (num_left / m) * calculate_total_loss(criterion, @view target_values[left_els]) + (num_right / m) * calculate_total_loss(criterion, @view target_values[.!left_els])
		gain = parent_loss - loss

		if gain > best_gain
			best_gain = gain
			best_left_set = left_set
		end
	end

	return (gain=best_gain, threshold=CategoricalSplitRule(best_left_set))
end

function best_split(
	data::AbstractDataFrame, 
	target_values::AbstractVector, 
	features::Vector{Symbol}, 
	criterion::Criterion
)
	best_gain = -Inf
	best_feature = :none
	best_threshold = nothing

	for feature in features
		feature_values = data[!, feature]

		result = eltype(feature_values) <: Real ? numerical_split(feature_values, target_values, criterion) :
		categorical_split(feature_values, target_values, criterion)

		if result.gain > best_gain
			best_gain = result.gain
			best_feature = feature
			best_threshold = result.threshold
		end

	end

	return (gain=best_gain, threshold=best_threshold, feature=best_feature)
end

split_threshold(rule::ThresholdSplitRule, value) = value <= rule.threshold
split_threshold(rule::CategoricalSplitRule, value) = value in rule.left_values

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

	threshold_split = split_threshold.(Ref(split.threshold), best_feature)

	# threshold_split = eltype(split.threshold) <: Real ? best_feature .<= split.threshold : 

	@show split.threshold

	left_split = @view data[threshold_split, :]
	right_split = @view data[.!threshold_split, :]

	# Recurse on subtrees
	left = fit_tree(left_split, target, features, criterion, minsplit, maxdepth, currentdepth + 1)
	right = fit_tree(right_split, target, features, criterion, minsplit, maxdepth, currentdepth + 1)

	return TNode(pred, current_error, split.feature, split.threshold, left, right)
end