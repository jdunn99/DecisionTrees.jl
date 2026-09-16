struct PruneEvent{T}
	alpha::Float64
	pruned_nodes::Vector{TNode{T}}
end

function prune!(node::TNode)
	node.is_leaf = true
	node.left = nothing
	node.right = nothing
end

function print_cp(node::TNode)
	root_error = node.error
	events = generate_alphas(node)

	cp = []

	for event in events
		push!(cp, event.alpha / root_error)
	end

	return cp
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
	subtrees = [deepcopy(current)]
	alphas = [0.0]
	pq = PriorityQueue{typeof(current), Float64}()

	for node in get_subtree(current)
		pq[node] = (node.error - node.subtree_error) / (node.number_leaves - 1)
	end

	while !isempty(pq) && current.number_leaves > 1
		best_alpha = first(pq)[2]

		same_alpha = []
		while !isempty(pq) && isapprox(first(pq)[2], best_alpha; atol=1)
			pair = popfirst!(pq)
			push!(same_alpha, pair[1])
		end

		pruned = TNode{T}[]
		for node in same_alpha
			if is_attached(node)
				prune!(node, pq)
				push!(pruned, deepcopy(node))
			end
		end

		push!(events, PruneEvent(best_alpha, pruned))
	end

	return events
end

# function generate_alphas(tree::TNode)
# 	subtrees = [deepcopy(tree)]
# 	alphas = [0.0]
# 	current = deepcopy(tree)

# 	while count_leaves(current) > 1
# 		subtree = get_subtree(current)
# 		node_alphas = Dict{TNode, Float64}()
# 		best_alpha = Inf
# 		weakest_node = nothing

# 		for node in subtree
# 			error = node.error
# 			sub_error = subtree_error(node)
# 			leaves = count_leaves(node)

# 			node_alphas[node] = (error - sub_error) / (leaves - 1)
# 		end

# 		# Handle matching alphas
# 		best_alpha = minimum(values(node_alphas))
# 		weakest_nodes = [node for node in subtree if isapprox(node_alphas[node], best_alpha; atol=1e-8)]

# 		for node in weakest_nodes
# 			prune!(node)
# 		end

# 		push!(subtrees, deepcopy(current))
# 		push!(alphas, best_alpha)
# 	end

# 	return (subtrees=subtrees, alphas=alphas)
# end
