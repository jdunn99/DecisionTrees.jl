"""
	PruneEvent{T}

Used to track the result of each pruning step.

# Fields
- `alpha`: The alpha used to prune
- `pruned_nodes`: The nodes removed from the tree
- `number_splits`: The new number of splits in the tree
- `rel_error`: The associated error of the newly pruned tree
"""
struct PruneEvent{T}
	alpha::Float64
	pruned_nodes::Vector{TNode{T}}
	number_splits::Int
	rel_error::Float64
end


"""
	prune!(node, pq)

Collapse `node` into a leaf in place. Resets the cached subtree from the PriorityQueue

NOTE: Mutates BOTH node and pq
"""
function prune!(node::TNode, pq::PriorityQueue)
	# Remove all removed nodes from the Priority Queue
	for n in get_subtree(node)
		haskey(pq, n) && delete!(pq, n)
	end

	node.is_leaf = true
	node.left = nothing
	node.right = nothing
	node.subtree_error = node.error
	node.feature = :none
	node.threshold = 0.0
	node.number_leaves = 1

	update_tree!(node, pq)
	return nothing
end

"""
	update_tree!(node, pq)

Helper function that walks from a `node` to the root and recalculates the `subtree_error` & `number_leaves`
from its children. Updating the `alpha` in `pq`.
"""
function update_tree!(node::TNode, pq::PriorityQueue)
	parent = node.parent
	while parent !== nothing
		parent.subtree_error = parent.left.subtree_error + parent.right.subtree_error
		parent.number_leaves = parent.left.number_leaves + parent.right.number_leaves

		# Update the parent node with the new alpha after pruning
		if haskey(pq, parent)
			pq[parent] = (parent.error - parent.subtree_error) / (parent.number_leaves - 1)
		end
		parent = parent.parent
	end
	return nothing
end

"""
	prune_weakest_nodes!(pq)

Pop and prune every node tied with the current minimum alpha in `pq` skipping any that were
already pruned by an ancestor (see [`is_attached`]@ref)

# Returns
`nothing` if `pq` is empty. Otherwise `(alpha, pruned)` where `pruned` is a `Vector` of deep-copied nodes that were just pruned
"""
function prune_weakest_nodes!(pq::PriorityQueue{TNode{T}, Float64}) where {T}
	isempty(pq) && return nothing
	best_alpha = first(pq)[2]
	pruned = TNode{T}[]

	# Remove and prune all nodes sharing the same minimum alpha
	while !isempty(pq) && isapprox(first(pq)[2], best_alpha; atol=1e-10)
		node, _ = popfirst!(pq)
		# push!(same_alpha, pair[1])

		if is_attached(node)
			prune!(node, pq)
			push!(pruned, deepcopy(node))
		end
	end

	return (alpha=best_alpha, pruned=pruned)
end


"""
	prune_to_target(tree, target_alpha)

Prunes a tree to a given level of alpha.
Collapses the weakest link via [`prune_weakest_nodes!`](@ref) as long as next weakest alpha is
`<= target_alpha`

# Fields
- `tree`: The tree being pruned.
- `target_alpha`: The target alpha to prune the tree to.

Does not mutate the tree. Returns a `deepcopy` so the original is safe to reuse.o

# Examples
```jldoctest
julia> using DecisionTrees, DataFrames

julia> data = DataFrame(x = [1.0, 2.0, 3.0, 4.0], y = [0.0, 0.0, 1.0, 1.0]);

julia> tree = fit_tree(data, :y, [:x], MSECriterion(); minsplit=1, maxdepth=2);

julia> pruned = prune_to_target(tree, Inf);  # prune everything

julia> pruned.is_leaf
true
```
"""
function prune_to_target(tree::TNode{T}, target_alpha::Float64) where {T}
	current = deepcopy(tree)
	pq = build_queue(current)

	while !isempty(pq) && first(pq)[2] <= target_alpha
		prune_weakest_nodes!(pq)
	end

	return current
end