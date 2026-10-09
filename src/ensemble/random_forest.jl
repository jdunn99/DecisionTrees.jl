function random_split(
	data::AbstractDataFrame, 
	target_values::AbstractVector, 
	features::Vector{Symbol}, 
	criterion::Criterion,
	mtry::Union{Int, Nothing}=nothing
)
	n = length(feautres)

	if mtry === nothing || mtry >= n
		return best_split(data, target_values, features, criterion)
	end

	indices = randperm(n)
	selected_features = features[indices[1:mtry]]
	split = best_split(data, target_values, selected_features, criterion)

	if split.gain != -Inf
        return split
    else
    	selected_features = features[indices[(mtry + 1):end]]
        return best_split(data, target_values, selected_features, criterion)
    end
end

function fit()
end