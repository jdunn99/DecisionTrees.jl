# MAJOR TODO: We need to handle categorical data as a feature in regression.

abstract type Criterion end

abstract type ClassificationCriterion <: Criterion end
struct GiniCriterion <: ClassificationCriterion end

abstract type RegressionCriterion <: Criterion end
struct MSECriterion <: RegressionCriterion end

function unique_classes(labels::AbstractVector)
	classes = Dict{eltype(labels), Int}()
	for class in labels
		classes[class] = get(classes, class, 0) + 1
	end

	return classes
end

"""
1. Count the unique classes in the target labels
2. p = (count / length)

"""
function classification_probability(labels::AbstractVector)
	n = length(labels)

	# Exception (?)
	n == 0 && return Dict{eltype(labels), Float64}()

	classes = unique_classes(labels)
	return Dict(k => v / n for (k, v) in classes)
end

# Calculate Gini loss
function calculate_loss(::GiniCriterion, labels::AbstractVector)
	total_p = 0
	probabilities = classification_probability(labels)

	for p in values(probabilities)
		total_p += p^2
	end

	return 1.0 - total_p
end

# Calculate MSE loss
function calculate_loss(::MSECriterion, values::AbstractVector)
	n = length(values)
	n == 0 && return -Inf

	μ = sum(values) / n
	s = 0.0

	for val in values
		s += (val - μ)^2
	end

	return s / n
end