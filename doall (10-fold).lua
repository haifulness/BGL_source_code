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
	max_epoch = 1e7,  -- max number of updates
}

optimState = {
	learningRate = 1e-5
	--, momentum = 1e-4
	, weightDecay = 1e-5
	, learningRateDecay = 1e-8
}

torch.manualSeed(os.clock())
--torch.setnumthreads(params.threads)


-------------------------------------------------------------------------------
-- Run
--
-- Load files
dofile "1-data.lua"
load_data("data/data.txt")
local data, target = gen_data()

dofile "2-model.lua"
model, criterion = buildModel(1, 1, 1, 1, "sigmoid")

dofile "3-train.lua"

-- Build models and train
-- Number of hidden layers and hidden nodes are considered based on
-- number of input (23) and number of samples (305)
--   Layers: 2 -> 5
--   Nodes: 20 -> 25
for num_hidden_layers = 2, 5 do
	for num_hidden_nodes = 20, 25 do
		for turn = 1, 5 do
			print("\nNum of hidden layers: ", num_hidden_layers, "\nNum of hidden nodes: ", num_hidden_nodes, "\nTurn: ", turn)

			local log, logErr = io.open("log.txt", "a+")
			if logErr then 
				print("File open error")
				break
			end

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
			local train_plot, test_plot = {}, {}

			for fold = 1, NUM_FOLDS do
				train_plot[fold], test_plot[fold] = {}, {}

				-- Build model
				model, criterion = buildModel(INPUT_SIZE, num_hidden_nodes, 1, num_hidden_layers, "sigmoid")
				-- Create train & test datasets
				local train_input = torch.Tensor(DATASET_SIZE - #idx[fold], INPUT_SIZE)
				local train_output = torch.Tensor(DATASET_SIZE - #idx[fold])
				local test_input = torch.Tensor(#idx[fold], INPUT_SIZE)
				local test_output = torch.Tensor(#idx[fold])

				-- Pull data into the train and test sets
				for i = 1, NUM_FOLDS do
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

				-- Train
				for epoch = 1, params.max_epoch do
					train_err[fold] = train(train_input, train_output)
					test_err[fold] = test(test_input, test_output)

					-- Early Stopping
					

					if train_err[fold] ~= train_err[fold] then
						break
					end
					
					table.insert(train_plot[fold], train_err[fold])
					table.insert(test_plot[fold], test_err[fold])
				end			
			end

			-- Calculate the avg error
			--[
			local total_test_err = 0
			local not_nan_counter = 0
			for fold = 1, NUM_FOLDS do
				if test_err[fold] ~= test_err[fold] then
					-- do nothing
				else
					total_test_err = total_test_err + test_err[fold]
					not_nan_counter = not_nan_counter + 1
				end
			end
			
			print("Average Test Error ", total_test_err/not_nan_counter)

			-- Log
			--
			log:write("\n-------------------------------------------\n")
			log:write("\nNum of hidden layers: ", num_hidden_layers)
			log:write("\nNum of hidden nodes: ", num_hidden_nodes)
			log:write("\nTurn: ", turn)
			log:write("\nTrain & Test Error: ")
			for fold = 1, NUM_FOLDS do
				log:write("\n" .. train_err[fold] .. ", " .. test_err[fold])
			end
			log:write("\nAverage Test Error: ", total_test_err/not_nan_counter)
			log:write("\nTime: ", timer:time().real)
			log:close()

			print("Total time: " .. timer:time().real .. " seconds\n")

			-- Plot
			--
			local graphFile = 'graph_' .. num_hidden_layers .. '_' .. num_hidden_nodes .. '_' .. turn .. '_' .. '.png'
			gnuplot.pngfigure(graphFile)
			gnuplot.title('Training & Test loss')
			gnuplot.ylabel('Loss')
			gnuplot.xlabel('Epoch')
			gnuplot.plot(
				{'Training Loss', torch.Tensor(train_plot[1])}
				, {'Test Loss', torch.Tensor(test_plot[1])}
			)
			gnuplot.plotflush()
		end
	end
end
