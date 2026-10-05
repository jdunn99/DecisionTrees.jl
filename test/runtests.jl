using DecisionTrees
using Test
using Plots

path = "C:/Users/jack/Desktop/Credit Default.csv"
data = DecisionTrees.load_data(path)

# target_class = :species
# features_class = [:petal_length, :petal_width, :sepal_length]

target_reg = :petal_length
features_reg = [:petal_width, :sepal_length, :sepal_width]

target_class = :credit_default
features_class = [:BILL_AMT1, :BILL_AMT2, :BILL_AMT3, :BILL_AMT4, :BILL_AMT5, :BILL_AMT6,
            :PAY_AMT1, :PAY_AMT2, :PAY_AMT3, :PAY_AMT4, :PAY_AMT5, :PAY_AMT6]

mse(actual, pred) = sum((actual .- pred).^2 / length(actual))

@testset "DecisionTrees.jl" begin
    model = DecisionTrees.fit(data.train_data, target_class, features_class, DecisionTrees.GiniCriterion())
    DecisionTrees.print_cp(model)

    @show model.tree.number_leaves

    plt = DecisionTrees.plot_tree(model.tree)
    savefig(plt, "base.png")

    test_cp = 0.001708
    alpha = test_cp * model.root_error
    pruned = DecisionTrees.prune_to_target(model.tree, alpha)

    plt = DecisionTrees.plot_tree(pruned)
    savefig(plt, "pruned.png")
end

# @testset "Bagging" begin
#     # actual = @view data.test_data[!, target_reg]
#     # tree = DecisionTrees.fit_tree(data.train_data, target_reg, features_reg, DecisionTrees.GiniCriterion())
#     # tree_prediction = DecisionTrees.predict(tree, data.test_data)

#     model = DecisionTrees.bootstrap(data.train_data, target_class, features_class, DecisionTrees.GiniCriterion(), 25)
#     @show bagging_predictions = DecisionTrees.predict_ensemble(model, data.test_data)


#     # @show mse(actual, tree_prediction)
#     # @show mse(actual, bagging_predictions)
# end