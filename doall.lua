--[[
-- Author: Hai Tran
-- Date: May 22, 2016
-- File: doall.lua
--]]

require "optim"
require "nn"
require "gnuplot"
require("CustomLinear.lua")


-------------------------------------------------------------------------------
-- Constants.
--
INPUT_SIZE = 23
DATASET_SIZE = 305

-- k-fold cross-validation
NUM_FOLDS = 10

-- Size of each subset
SUB_SIZE = math.floor(DATASET_SIZE / NUM_FOLDS)


-------------------------------------------------------------------------------
-- Configuration
params = {
	seed = 1,  -- initial random seed
	threads = 1,  -- number of threads
	beta = 1e1,  -- prediction error coefficient
	batch_size = 1,  -- batch size
	max_epoch = 1e4,  -- max number of updates
}

optimState = {
	learningRate = 1e-3
	--, momentum = 1e-3
	--, weightDecay = 1e-3
	, learningRateDecay = 1e-1
}

torch.manualSeed(os.clock())
--torch.setnumthreads(params.threads)


-------------------------------------------------------------------------------
-- Run
--
-- Load files
dofile "1-data.lua"
dofile "2-model.lua"
dofile "3-train.lua"

-- Load data
load_data("data/data.txt")
local data, target = gen_data()

-- Build models and train
for num_hidden_layers = 3, 3 do
	for num_hidden_nodes = 5, 5 do
		local timer = torch.Timer()

		-- Prepare data
		local indices = torch.randperm(DATASET_SIZE)
		local idx = {}

		-- Assign indices into subsets
		for i = 1, NUM_FOLDS do
			idx[i] = {}
			for j = 1, SUB_SIZE do
				idx[i][j] = indices[(i-1) * SUB_SIZE + j]
			end
		end

		-- The last subset receives the remains
		for j = 1, DATASET_SIZE % NUM_FOLDS do
			idx[NUM_FOLDS][SUB_SIZE + j] = indices[DATASET_SIZE - j + 1]
		end

		local train_err, test_err = {}, {}

		for fold = 1, NUM_FOLDS do
			-- Build model
			local model = buildModel(INPUT_SIZE, num_hidden_nodes, 1, num_hidden_layers, "sigmoid")
			-- Create train & test datasets
			local train_input = torch.Tensor(DATASET_SIZE - #idx[fold], INPUT_SIZE)
			local train_output = torch.Tensor(DATASET_SIZE - #idx[fold])
			local test_input = torch.Tensor(#idx[fold], INPUT_SIZE)
			local test_output = torch.Tensor(#idx[fold])

			-- Pull data into the train and test sets
			for i = 1, 10 do
				local train_counter = 1

				if i == fold then
					for j = 1, #idx[i] do
						test_input[j] = data[idx[i][j]]
						test_output[j] = target[idx[i][j]]
					end
				else
					for j = 1, #idx[i] do
						for k = 1, INPUT_SIZE do
							train_input[train_counter][k] = data[idx[i][j]][k]
						end
						train_output[train_counter] = target[idx[i][j]]
						train_counter = train_counter + 1
					end
				end
			end

			for epoch = 1, params.max_epoch do
				train_err[fold] = train(model, train_input, train_output)

				if train_err[fold] ~= train_err[fold] 
				or train_err[fold] > -math.huge 
				or train_err[fold] < math.huge 
				then 
					train_err[fold] = train(model, train_input, train_output)
				end

				if epoch % 1e3 == 0 then
					print(fold, epoch, train_err[fold])
				end
			end

			test_err[fold] = test(model, test_input, test_output)
			
			print(fold, train_err[fold], test_err[fold])
			
		end

		-- Calculate the avg error
		--[
		local total_err = 0
		for fold = 1, NUM_FOLDS do
			total_err = total_err + test_err[fold]
		end
		print("Average Test Error ", total_err/NUM_FOLDS)
		--]]

		print("Total time: " .. timer:time().real .. " seconds\n")
	end
end
