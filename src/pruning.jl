struct PruneEvent{T}
	alpha::Float64
	pruned_nodes::Vector{TNode{T}}
	number_splits::Int
	rel_error::Float64
end

function prune!(node::TNode)
	node.is_leaf = true
	node.left = nothing
	node.right = nothing
end

function print_cp(node::TNode)
    root_error = node.error
    events = generate_alphas(node)
    
    println("\nComplexity Parameter Table:")
    
    header = rpad("CP", 10) * rpad("nsplit", 10) * rpad("rel error", 10)
    println(header)
    println("-" ^ length(header))

    for event in reverse(events)
        cp_val = event.alpha / root_error
        
        cp_val = max(0.0, cp_val) 
        
        cp_str = rpad(round(cp_val, digits=4), 10)
        nsplit_str = rpad(event.number_splits, 10)
        rel_error_str = rpad(round(event.rel_error, digits=4), 10)
        
        println(cp_str * nsplit_str * rel_error_str)
    end

    # Return just the raw CP values
    return [e.alpha / root_error for e in events]
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

function generate_alphas(tree::TNode{T}) where {T}
	current = deepcopy(tree)
	events = PruneEvent{T}[] # Used for tree reconstruction
	pq = PriorityQueue{typeof(current), Float64}()

	# Base state
	root_error = current.error
	push!(events, PruneEvent(0.0, TNode{T}[], current.number_leaves - 1, current.subtree_error / root_error))

	# Store alphas of root nodes in order
	for node in get_subtree(current)
		if node.number_leaves > 1
			pq[node] = (node.error - node.subtree_error) / (node.number_leaves - 1)
		end
	end

	while !isempty(pq) && current.number_leaves > 1
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

		number_splits = current.number_leaves - 1
		rel_error = current.subtree_error / root_error

		push!(events, PruneEvent(best_alpha, pruned, number_splits, rel_error))
	end

	return events
end