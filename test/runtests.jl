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

    DecisionTrees.fit_tree(data.train_data, :petal_length, [:species, :sepal_length, :sepal_width, :petal_width], DecisionTrees.MSECriterion())

    # model = DecisionTrees.fit(data.train_data, target_class, features_class, (minsplit=10,maxdepth=5,k=5))
    # DecisionTrees.print_cp(model)

end
