function mean(x::AbstractVector{<:Real})
	n = length(x)
	n == 0 && return 0.0

	return sum(x) / n
end

function std_dev(x::AbstractVector{<:Real})
	n = length(x)
	n <= 1 && return 0.0

	m = mean(x)
	ss = sum((v - m)^2 for v in x)
	return sqrt(ss / (n - 1))
end