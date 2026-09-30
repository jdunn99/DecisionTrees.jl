using DecisionTrees
using Test
using Plots

@testset "DecisionTrees.jl" begin
    # Write your tests here.
    path = "C:/Users/jack/Desktop/iris.csv"
    data = DecisionTrees.load_data(path)

    target_class = :species
    features_class = [:petal_length, :petal_width, :sepal_length]

    target_reg = :petal_length
    features_reg = [:petal_width, :sepal_length, :sepal_width]

    # tree = DecisionTrees.fit_tree(data.train_data, :petal_length, [:species, :sepal_length, :sepal_width, :petal_width], DecisionTrees.MSECriterion())

    model = DecisionTrees.fit(data.train_data, target_class, features_class, DecisionTrees.GiniCriterion())
    # DecisionTrees.print_cp(model)

    plt = DecisionTrees.plot_tree(model.tree)
    savefig(plt, "base.png")

    test_cp = 0.014925
    alpha = test_cp * model.root_error
    pruned = DecisionTrees.prune_to_target(model.tree, alpha)

    plt = DecisionTrees.plot_tree(pruned)
    savefig(plt, "pruned.png")
end
