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

tree_error(::ClassificationCriterion, actual::AbstractVector, pred) =
	Float64(count(!=(pred), actual))
tree_error(::RegressionCriterion, actual::AbstractVector, pred::Real) =
	sum(x -> (x - pred)^2, actual)
tree_error(::ClassificationCriterion, actual::AbstractVector, predicted::AbstractVector) =
	Float64(sum(actual .!= predicted))
tree_error(::RegressionCriterion, actual::AbstractVector, predicted::AbstractVector) =
	sum((actual .- predicted).^2)

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