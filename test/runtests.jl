using DecisionTrees
using Test

@testset "DecisionTrees.jl" begin
    # Write your tests here.
    path = "C:/Users/jack/Desktop/iris.csv"
    data = DecisionTrees.load_data(path)

    target_class = :species
    features_class = [:petal_length, :petal_width, :sepal_length]

    target_reg = :petal_length
    features_reg = [:petal_width, :sepal_length, :sepal_width]

    tree = DecisionTrees.fit_tree(data.train_data, target_reg, features_reg, DecisionTrees.MSECriterion())
    pred = DecisionTrees.predict(tree, data.test_data)

    @show tree.error

    # pruning_result = DecisionTrees.generate_alphas(tree)
    cp = DecisionTrees.print_cp(tree)

    @show cp
end
