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

