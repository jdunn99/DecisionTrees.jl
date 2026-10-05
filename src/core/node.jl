"""
	tree_error(criterion, actual, pred)

Total error of prediction `pred` for every element of `actual` under the given `criterion`.

Accepts a single scalr `pred` or a vector `predicted`.
Calculation counts missclassifications, regression sums squared error.
"""
tree_error(::ClassificationCriterion, actual::AbstractVector, pred) =
	Float64(count(!=(pred), actual))
tree_error(::RegressionCriterion, actual::AbstractVector, pred::Real) =
	sum(x -> (x - pred)^2, actual)
tree_error(::ClassificationCriterion, actual::AbstractVector, predicted::AbstractVector) =
	Float64(sum(actual .!= predicted))
tree_error(::RegressionCriterion, actual::AbstractVector, predicted::AbstractVector) =
	sum((actual .- predicted).^2)

"""
	is_attached(node)

Check whether a `node` is still reachable from its parent (checks if the node has been pruned).

Pruning ancestors detaches descendants which must be skipped instead of repruned.
"""
is_attached(node::TNode) = node.parent === nothing || node.parent.left === node || node.parent.right === node

"""
	get_subtree(node)

Gets the internal subtree nodes via pre-order tree traversal.
Used in building list of alphas in pruning.
"""
function get_subtree(node::TNode)
	nodes = TNode[]
	_get_subtree!(nodes, node)
	return nodes
end

"""
	_get_subtree!(nodes, node)

Recursion helper function for `get_subtree`. Addes a `node` to `nodes` if it's not a leaf
then recurses on their children.
"""
function _get_subtree!(nodes::Vector{TNode}, node::TNode)
	if !node.is_leaf
		Base.push!(nodes, node)
		_get_subtree!(nodes, node.left)
		_get_subtree!(nodes, node.right)
	end
	return nothing
end

function tree_path(node::TNode)
	positions = Dict{TNode, Tuple{Float64, Float64}}()
	trace_path!(node, 0, Ref(0), positions)
	return positions
end

# TODO: Fix any type
function draw_edges!(node::TNode, plt, positions)
	node.is_leaf && return
	x0, y0 = positions[node]
	for child in (node.left, node.right)
		x1, y1 = positions[child]
		plot!(plt, [x0, x1], [y0, y1], linecolor=:gray)
		draw_edges!(child, plt, positions)
	end
end

function draw_nodes!(node::TNode, plt, positions)
	x, y = positions[node]
	# TODO: Categorical threshold needs another condition

	label = node.is_leaf ?
			"pred=$(node.prediction))\nn=$(node.number_leaves)" :
			"$(node.feature)\nn=$(node.number_leaves) < $(node.threshold isa ThresholdSplitRule ? node.threshold.threshold : "")"
	color = node.is_leaf ? :lightgreen : :lightblue
	annotate!(plt, x, y, text(label, 8, :center))
	scatter!(plt, [x], [y], markersize=25, markershape=:rect, markercolor=color)

	node.is_leaf && return
	draw_nodes!(node.left, plt, positions)
	draw_nodes!(node.right, plt, positions)
end

function trace_path!(node::TNode, depth::Int, x_counter::Ref{Int}, positions::Dict)
	if node.is_leaf
		x = x_counter[]
		x_counter[] += 1
		positions[node] = (Float64(x), -Float64(depth))
		return x
	end

	#  If a split follow the split and get the x location of the split
	left = trace_path!(node.left, depth + 1, x_counter, positions)
	right = trace_path!(node.right, depth + 1, x_counter, positions)
	x = (left + right) / 2
	positions[node] = (x, -Float64(depth))

	return x
end

function plot_tree(tree::TNode)
	positions = tree_path(tree)

	n_leaves = tree.number_leaves
	max_depth = maximum(-y for (x, y) in values(positions))  # y is negative depth

	@show n_leaves, max_depth

	width = round(Int, n_leaves * 90)
	height = round(Int, (max_depth + 1) * 90) 

	plt = plot(legend=false, axis=false, grid=false, ticks=false, size=(width, height))

	draw_edges!(tree, plt, positions)
	draw_nodes!(tree, plt, positions)

	return plt
end