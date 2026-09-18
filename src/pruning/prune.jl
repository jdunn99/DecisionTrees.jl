struct PruneEvent{T}
	alpha::Float64
	pruned_nodes::Vector{TNode{T}}
	number_splits::Int
	rel_error::Float64
end

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

function prune_to_target(tree::TNode{T}, target_alpha::Float64) where {T}
	current = deepcopy(tree)
	pq = build_queue(current)

	while !isempty(pq) && first(pq)[2] <= target_alpha
		prune_weakest_nodes!(pq)
	end

	return current
end