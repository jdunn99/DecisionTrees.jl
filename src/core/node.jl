"""
	TNode{T}

A single node in a decision tree. Linked via `parent` nodes to support weakest-link pruning without rewalking.

# Fields
- `is_leaf::Bool`: whether the node is a predictor (`true`) or a split (`false`)
- `prediction::T`: The resulting prediction of the node if it is considered a leaf.
- `error::Float64`: This node's error. Used in pruning.
- `feature::Symbol`: The feature used in splitting the splitting. `:none` for leaves.
- `threshold::Float64`: The split threshold.
- `left`, `right`: Child nodes. `nothing` for leaves.
- `parent`: The node's parent node. `nothing` for the tree's root.
- `subtree_error::Float64`: Total error of all leaves below this node. Stored in priority queue cache
- `number_leaves::Int`: Number of leaves below this node.
"""
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

	"""
		TNode(pred, error)
		
	Construct a leaf node with no children
	"""
	TNode(pred::T, err::Float64) where {T} = new{T}(true, pred, 
			err, :none, 0.0, nothing, nothing, nothing, err, 1)

	"""
		TNode(pred, err, thres, l, r)

	Construct a splitting node from two children, computing `subtree_error` and `number_leaves`
	and updating the child node to reference the new parent.
	"""
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
		push!(nodes, node)
		_get_subtree!(nodes, node.left)
		_get_subtree!(nodes, node.right)
	end
	return nothing
end