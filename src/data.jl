# This is a temporary file used for loading and reading some data while I build out the package. 
# This is exclusively used in the test suite. This file will not be included once the package is fully developed.
function load_data(path)

	# Load data into frame
	sample_data = CSV.read(path, DataFrame)

	# Split into test and training sets based on random index array
	n = nrow(sample_data)
	sample_size = floor(Int, 0.70 * n)
	indices = randperm(n)

	train_indices = @view indices[1:sample_size]
	test_indices = @view indices[sample_size+1:end]

	train_data = @view sample_data[train_indices, :]
	test_data = @view sample_data[test_indices, :]

	return (test_data=test_data, train_data=train_data)
end
