# MAJOR TODO: We need to handle categorical data as a feature in regression.
# criterion.jl - Core classification and regression loss metrics 

"""
	Criterion

Abstract type for all spliting criteria used by [`best_split`](@ref) and [`fit_tree`](@ref)

Every subtype must have a corresponding `calculate_loss(::Criterion, values::AbstractVector)` method.
"""
abstract type Criterion end


"""
	ClassificationCriterion <: Criterion

Abstract type for splitting criteria when the target is categorical
"""
abstract type ClassificationCriterion <: Criterion end


"""
	GiniCriterion <: ClassificationCriterion

Categorical criterion corresponding to the Gini index.
"""
struct GiniCriterion <: ClassificationCriterion end


"""
	RegressionCriterion <: Criterion

Abstract type for splitting criteria when the target is numerical.
"""
abstract type RegressionCriterion <: Criterion end

"""
	MSECriterion <: RegressionCriterion

Categorical criterion corresponding to the Mean Squared Error.
"""
struct MSECriterion <: RegressionCriterion end


"""
	unique_classes(labels)

Calculate the number of unique classes among labels and count their frequency.

# Arguments
- `labels`: Target labels of classification results

# Returns
`classes` Dictionary mapping unique class to frequency in labels.
	
"""
function unique_classes(labels::AbstractVector)
	classes = Dict{eltype(labels), Int}()
	for class in labels
		classes[class] = get(classes, class, 0) + 1
	end

	return classes
end


"""
	classification_probability(labels)

Calculates the probability of each unique class.
p = (count / length)

# Arguments
- `labels`: Target labels of classification results

# Returns
`probabilities` Dictionary mapping class to overall probability
"""
function classification_probability(labels::AbstractVector)
	n = length(labels)

	# Exception (?)
	n == 0 && return Dict{eltype(labels), Float64}()

	classes = unique_classes(labels)
	probabilities = Dict(k => v / n for (k, v) in classes)

	return probabilities
end

"""
	calculate_loss(::GiniCriterion, labels)

Calculates the gini impurity of a split relative to each unique class.
Gini impurity is defined as 1 - ∑ (p_i)^2.

# Arguments
- `labels`: Target classification labels

# Returns
`loss` The total gini immpurity of the split
"""
# Calculate Gini loss
function calculate_loss(::GiniCriterion, labels::AbstractVector)
	total_p = 0
	probabilities = classification_probability(labels)

	for p in values(probabilities)
		total_p += p^2
	end

	loss = 1.0 - total_p

	return loss
end

"""
	calculate_loss(::MSECriterion, values::AbstractVector)

Calculates the Mean Squared Error (MSE) loss of a split relative to the target values.

# Arguments
- `values`: Target values

# Returns
`mse` the resulting MSE
"""
function calculate_loss(::MSECriterion, values::AbstractVector)
	n = length(values)
	n == 0 && return Inf

	μ = sum(values) / n
	s = 0.0

	for val in values
		s += (val - μ)^2
	end

	mse = s / n

	return mse
end