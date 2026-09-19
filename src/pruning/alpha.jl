"""
	build_queue(tree)

Builds a `PriorityQueue` of tree alphas
"""
function build_queue(tree::TNode{T}) where {T}
	pq = PriorityQueue{TNode{T}, Float64}()

	for node in get_subtree(tree)
		if node.number_leaves > 1
			pq[node] = (node.error - node.subtree_error) / (node.number_leaves - 1)
		end
	end

	return pq
end


"""
	generate_alphas(tree)

Computer the full cost complexity pruning sequence, returning a `PruneEvent` per each alpha breakpoint.
Does not mutate the original `tree`. Used to build the alpha/cp table.
"""
function generate_alphas(tree::TNode{T}) where {T}
	current = deepcopy(tree)
	events = PruneEvent{T}[] # Used for tree reconstruction
	root_error = current.error

	push!(events, PruneEvent(0.0, TNode{T}[], current.number_leaves - 1, current.subtree_error / root_error))

	pq = build_queue(current)

	while !isempty(pq) && current.number_leaves > 1
		result = prune_weakest_nodes!(pq)
		result === nothing && break

		pruned = [deepcopy(subtree) for subtree in result.pruned]
		number_splits = current.number_leaves - 1
		rel_error = current.subtree_error / root_error

		push!(events, PruneEvent(result.alpha, pruned, number_splits, rel_error))
	end

	return events
end

