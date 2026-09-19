"""
	prediction(criterion, target_values)

Calculate the value of a leaf given `target_values`.
Majority class for classification. Mean for regression
"""
prediction(::ClassificationCriterion, target_values::AbstractVector) = argmax(unique_classes(target_values))
prediction(::RegressionCriterion, target_values::AbstractVector) = round(sum(target_values) / length(target_values), digits=4)

"""
	predict_row(node, row)

Helper function to recurse through a tree and make a prediction.
"""
function predict_row(node::TNode, row)
	node.is_leaf && return node.prediction

	if row[node.feature] <= node.threshold
		return predict_row(node.left, row)
	else
		return predict_row(node.right, row)
	end
end

"""
	predict(tree, data)
	predict(tree, observation)

Predict from a fitted tree. 
Given an `AbstractDataFrame` returns a vector of predictions.
Given a single `NamedTuple` observation, returns one prediction

# Examples
```jldoctest
julia> using DecisionTrees, DataFrames

julia> data = DataFrame(x = [1.0, 2.0, 3.0, 4.0], y = [0.0, 0.0, 1.0, 1.0]);

julia> tree = fit_tree(data, :y, [:x], MSECriterion());

julia> predict(tree, data)
4-element Vector{Float64}:
 0.0
 0.0
 1.0
 1.0

julia> predict(tree, (x = 2.5))
0.0
```
"""
function predict(tree::TNode, data::AbstractDataFrame)
	return [predict_row(tree, row) for row in eachrow(data)]
end
predict(tree::TNode, observation::NamedTuple) = predict_row(tree, observation)

