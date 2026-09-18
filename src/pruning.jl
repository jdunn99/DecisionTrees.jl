""" Structs """
struct PruneEvent{T}
	alpha::Float64
	pruned_nodes::Vector{TNode{T}}
	number_splits::Int
	rel_error::Float64
end

""" Printing & Metrics """

""" K-Fold CV"""
function kfold_splits(n::Int, k::Int)
	quotient, remainder = divrem(n, k)
	idx = randperm(n)
	folds = Vector{Vector{Int}}(undef, k)

	# evenly split into k-sized folds
	fold_sizes = [quotient + (i <= remainder ? 1 : 0) for i in 1:k]

	start = 1
	for i in 1:k
		stop = start + fold_sizes[i]
		folds[i] = idx[start:stop-1]
		start = stop
	end

	@show folds
	return folds
end

function cross_validate(
	data::AbstractDataFrame,
	target::Symbol,
	features::Vector{Symbol},
	criterion::Criterion,
	events::Vector{<:PruneEvent},
	root_error::Float64,
	k::Int = 10,
	minsplit::Int = 10,
	maxdepth::Int = 5,
)
	n = nrow(data)
	alphas = [event.alpha for event in events]
	m = length(alphas)
	folds = kfold_splits(n, k)
	thresholds = Vector{Float64}(undef, m)

	# Guarantee that threshold values are within [α_i, α_i+1]
	for i in 1:(m - 1)
		thresholds[i] = sqrt(alphas[i] * alphas[i + 1])
	end
	# Guarantee full pruning for last alpha
	thresholds[m] = alphas[m] + thresholds[m-1]

	# error at (alpha size, fold)
	errors = zeros(Float64, m, k)

	for(i, test_indices) in enumerate(folds)
		@show (i, test_indices)

		# Build tree on k-1 folds
		train_indices = setdiff(1:n, test_indices)
		tree = fit_tree(data[train_indices, :], target, features, criterion, minsplit, maxdepth)

		# Independently prune trees at each threshold and test on kth fold
		for j in 1:m
			pruned = prune_to_target(tree, thresholds[j])
			preds = predict(pruned, data[test_indices, :])
			errors[j, i] = round(tree_error(criterion, data[test_indices, target], preds), digits=4)
		end
	end

	total_error = vec(sum(errors, dims=2))
	xerror = total_error ./ root_error
	xstd = [std_dev((errors ./ root_error)[i,:]) / sqrt(k) for i in 1:m]

	return (thresholds=thresholds, xerror=xerror, xstd=xstd)
end

""" Alpha Gen and Node Pruning """

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

function build_queue(tree::TNode{T}) where {T}
	pq = PriorityQueue{TNode{T}, Float64}()

	for node in get_subtree(tree)
		if node.number_leaves > 1
			pq[node] = (node.error - node.subtree_error) / (node.number_leaves - 1)
		end
	end

	return pq
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

function prune_to_target(tree::TNode{T}, target_alpha::Float64) where {T}
	current = deepcopy(tree)
	pq = build_queue(current)

	while !isempty(pq) && first(pq)[2] <= target_alpha
		prune_weakest_nodes!(pq)
	end

	return current
end