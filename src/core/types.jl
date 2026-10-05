# Criterion: How to score a split

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

# CriterionState: Running statistics to incrementally score a split.

abstract type CriterionState end
abstract type ClassificationState <: CriterionState end
abstract type RegressionState <: CriterionState end

"""
    GiniState{T}

Running class counts for [`GiniCriterion`](@ref).
"""
mutable struct GiniState{T} <: ClassificationState
	n::Int
	counts::Dict{T, Int}
end
GiniState{T}() where {T} = GiniState{T}(0, Dict{T,Int}())

mutable struct MSEState <: RegressionState
	n::Int
	sum::Float64
	sumsq::Float64
end
MSEState() = MSEState(0, 0.0, 0.0)

# SplitRule: Contains the rule used to determine which way to split
abstract type SplitRule end

struct ThresholdSplitRule <: SplitRule
	threshold::Float64
end

struct CategoricalSplitRule{T} <: SplitRule
	left_values::Set{T}
end

# TNode: A tree node

"""
	TNode{T}

A single node in a decision tree. Linked via `parent` nodes to support weakest-link pruning without rewalking.

# Fields
- `is_leaf::Bool`: whether the node is a predictor (`true`) or a split (`false`)
- `prediction::T`: The resulting prediction of the node if it is considered a leaf.
- `error::Float64`: This node's error. Used in pruning.
- `feature::Symbol`: The feature used in splitting the splitting. `:none` for leaves.
- `threshold::Float64`: The split threshold.
- `left`, `right`: Child nodes. `nothing` for leaves.
- `parent`: The node's parent node. `nothing` for the tree's root.
- `subtree_error::Float64`: Total error of all leaves below this node. Stored in priority queue cache
- `number_leaves::Int`: Number of leaves below this node.
"""
mutable struct TNode{T} 
	is_leaf::Bool
	prediction::T
	error::Float64 # Used for pruning
	feature::Symbol
	threshold::Union{SplitRule, Nothing}
	left::Union{TNode{T}, Nothing}
	right::Union{TNode{T}, Nothing}
	parent::Union{TNode{T}, Nothing}
	subtree_error::Float64
	number_leaves::Int

	"""
		TNode(pred, error)
		
	Construct a leaf node with no children
	"""
	TNode(pred::T, err::Float64) where {T} = new{T}(true, pred, 
			err, :none, nothing, nothing, nothing, nothing, err, 1)

	"""
		TNode(pred, err, thres, l, r)

	Construct a splitting node from two children, computing `subtree_error` and `number_leaves`
	and updating the child node to reference the new parent.
	"""
	function TNode(pred::T,
	 	 err::Float64, 
	 	 feat::Symbol, 
	 	 thres::SplitRule, 
	 	 l::TNode{T}, 
	 	 r::TNode{T}
	 ) where {T}
		node = new{T}(false, pred, err, feat, thres, l, r, nothing, 
			l.subtree_error + r.subtree_error, l.number_leaves + r.number_leaves)
		l.parent = node
		r.parent = node
	end

end

# Pruning

"""
	PruneEvent{T}

Used to track the result of each pruning step.

# Fields
- `alpha`: The alpha used to prune
- `pruned_nodes`: The nodes removed from the tree
- `number_splits`: The new number of splits in the tree
- `rel_error`: The associated error of the newly pruned tree
"""
struct PruneEvent{T}
	alpha::Float64
	pruned_nodes::Vector{TNode{T}}
	number_splits::Int
	rel_error::Float64
end

abstract type Model end
abstract type EnsembleTreeModel{T} <: Model end

# Model
"""
	Model{T}

The result of fitting and cross-validating a decision tree via [`fit`](@ref).
Bundles the tree itself with everything needed to build a complexity parameter table.

# Fields
- `tree`: The unpruned base tree
- `criterion`: The [`Criterion`](@ref) used to build the tree
- `target`: The column being predicted
- `features`: The columns used in splitting
- `minsplit`, `maxdepth`: Stopping parameters used in building the tree
- `root_error`: Unpruned root error used to normalize `xerror`
- `n`: The number of training rows
- `events`: The [`PruneEvent`](@ref) result from [`generate_alphas`](@ref)
- `thresholds`, `xerror`, `xstd`: Per level cross validation results from [`cross_validate`](@ref)
"""
struct DecisionTreeModel{T} <: Model
	tree::TNode{T}
	criterion::Criterion
	target::Symbol
	features::Vector{Symbol}
	minsplit::Int
	maxdepth::Int
	root_error::Float64
	n::Int
	events::Vector{PruneEvent{T}}
	thresholds::Vector{Float64}
	xerror::Vector{Float64}
	xstd::Vector{Float64}
end

struct BaggingTreeModel{T} <: EnsembleTreeModel{T}
	trees::Vector{TNode{T}}
	criterion::Criterion
	target::Symbol
	features::Vector{Symbol}
	minsplit::Int
	maxdepth::Int
end
