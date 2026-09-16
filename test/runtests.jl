using DecisionTrees
using Test

@testset "DecisionTrees.jl" begin
    # Write your tests here.
    path = "C:/Users/jack/Desktop/iris.csv"
    data = DecisionTrees.load_data(path)

    target = :species
    features = [:petal_length, :petal_width, :sepal_length]

    tree = DecisionTrees.fit_tree(data.train_data, target, features, DecisionTrees.GiniCriterion())
    @show tree
end
