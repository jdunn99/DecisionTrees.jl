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

    model = DecisionTrees.fit(data.train_data, target_class, features_class, DecisionTrees.GiniCriterion(), 2, 20)

    plt = plot_cp
    savefig(plt, "plot.png")
end
