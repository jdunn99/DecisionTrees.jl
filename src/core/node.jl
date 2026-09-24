"""
	tree_error(criterion, actual, pred)

Total error of prediction `pred` for every element of `actual` under the given `criterion`.

Accepts a single scalr `pred` or a vector `predicted`.
Calculation counts missclassifications, regression sums squared error.
"""
tree_error(::ClassificationCriterion, actual::AbstractVector, pred) =
	Float64(count(!=(pred), actual))
tree_error(::RegressionCriterion, actual::AbstractVector, pred::Real) =
	sum(x -> (x - pred)^2, actual)
tree_error(::ClassificationCriterion, actual::AbstractVector, predicted::AbstractVector) =
	Float64(sum(actual .!= predicted))
tree_error(::RegressionCriterion, actual::AbstractVector, predicted::AbstractVector) =
	sum((actual .- predicted).^2)

"""
	is_attached(node)

Check whether a `node` is still reachable from its parent (checks if the node has been pruned).

Pruning ancestors detaches descendants which must be skipped instead of repruned.
"""
is_attached(node::TNode) = node.parent === nothing || node.parent.left === node || node.parent.right === node

"""
	get_subtree(node)

Gets the internal subtree nodes via pre-order tree traversal.
Used in building list of alphas in pruning.
"""
function get_subtree(node::TNode)
	nodes = TNode[]
	_get_subtree!(nodes, node)
	return nodes
end

"""
	_get_subtree!(nodes, node)

Recursion helper function for `get_subtree`. Addes a `node` to `nodes` if it's not a leaf
then recurses on their children.
"""
function _get_subtree!(nodes::Vector{TNode}, node::TNode)
	if !node.is_leaf
		Base.push!(nodes, node)
		_get_subtree!(nodes, node.left)
		_get_subtree!(nodes, node.right)
	end
	return nothing
end