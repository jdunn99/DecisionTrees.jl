# Random Forest is just bagging with the added mtry...
function random_forest(
	data::AbstractDataFrame,
	target::Symbol,
	features::Vector{Symbol},
	criterion::Criterion;
	num_trees::Int=25,
	minsplit::Int = 10,
	maxdepth::Int = 5,
	mtry::Union{Int, Nothing}=nothing,
)
	p = length(features)
	m = something(mtry, 1)

	if !(1 <= m <= p)
		# TODO: Add more verbose error handling in future error handling patch
		throw(ArgumentError("Invalid mtry value"))
	end

	return bag(data, target, features, criterion, minsplit, maxdepth, m)
end