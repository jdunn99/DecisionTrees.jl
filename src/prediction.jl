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

