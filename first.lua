--[[
-- Author: Hai Tran
-- Date: Nov 21, 2015
-- Filename: bgl_firstCombination.lua
-- Descriptiion: Simulate the first combination that Peter Kok suggested.
-- The first combination consists of:
--   + Glucose level (at the start of the interval)
--   + Short acting insulin (during the interval)
--   + Food intake (during the interval)
--   + Exercise (during the interval)
]]

require 'torch'
require 'nn'
require 'gnuplot'
require("bgl_dataLoading.lua")
require("bgl_generateSets.lua")


-- Load data
local path = "../Datasets/Peter Kok - Real data for predicting blood glucose levels of diabetics/data.txt"
local 
	morning_date, 
    morning_time, 
    morning_glucose, 
    morning_SAI,  -- short acting insulin 
    morning_LAI,  -- long acting insulin
    morning_food,
    morning_exercise, 
    morning_stress,
    
    afternoon_date, 
    afternoon_time, 
    afternoon_glucose, 
    afternoon_SAI,  -- short acting insulin 
    afternoon_LAI,  -- long acting insulin 
    afternoon_food, 
    afternoon_exercise, 
    afternoon_stress,

    evening_date, 
    evening_time, 
    evening_glucose, 
    evening_SAI,  -- short acting insulin  
    evening_LAI,  -- long acting insulin 
    evening_food, 
    evening_exercise, 
    evening_stress,

    night_date, 
    night_time, 
    night_glucose, 
    night_SAI,  -- short acting insulin  
    night_LAI,  -- long acting insulin 
    night_food, 
    night_exercise, 
    night_stress  

	= loadFile(path)


-- All the above tables have the same size, so it is better
-- to have a constant to represent it.
NUM_DAY = #morning_date


-- Feed forward + back propagation. Right now I'm not using this function
-- because it requires too many params.
function gradientUpgrade(model, input, output, criterion, learningRate)
    local prediction    = model:forward(input)
    local err           = criterion:forward(prediction, output)
    local gradCriterion = criterion:backward(prediction, output)

    model:zeroGradParameters()
    model:backward(input, gradCriterion)
    model:updateParameters(learningRate)

    print(string.format("%.4f", err))

    return err
end

-- Create a tensor for expected values
local expectation_storage = {}
for i = 1, NUM_DAY - 1 do
    expectation_storage[i] = morning_glucose[i + 1]
end
local expectation = torch.Tensor(expectation_storage)





-- Neural net model:
--   + Linear
--   + 4 inputs
--   + 1 hidden layer with 10 nodes
--   + 1 output
--
local EPOCH_TIMES = 10
local BATCH_SIZE = 100
local NUM_BATCHES = 1
local SIZE_INPUT = 4
local SIZE_HIDDEN_LAYER = 200
local SIZE_OUTPUT = 1
local learningRate = 0.01
local momentum = 0.9


-- Divide the dataset
local train, test, validation = generateSets(NUM_DAY - 1, 50, 25, 25)
function train:size() return #train end

-- Input and output of the neural net
local input  = torch.Tensor(#train, SIZE_INPUT)
local output = torch.Tensor(#train)
local input_storage  = {}
local output_storage = {}
local counter = 0

-- Load data into input & output
for key, val in pairs(train) do
    if (type(val) == 'number') then
        counter = counter + 1
        input_storage[counter] = {}

        input_storage[counter][1] = morning_glucose[val]
        input_storage[counter][2] = morning_SAI[val]
        input_storage[counter][3] = morning_food[val]
        input_storage[counter][4] = morning_exercise[val]
        output_storage[counter]   = morning_glucose[val+1]
    end
end

-- Convert input to a Tensor
input  = torch.Tensor(input_storage)
output = torch.Tensor(output_storage)


-- Layers
module_01 = nn.Linear(SIZE_INPUT, SIZE_HIDDEN_LAYER)
module_02 = nn.Linear(SIZE_HIDDEN_LAYER, SIZE_OUTPUT)

-- Add the layer(s) to the net.
local net = nn.Sequential()
net:add(module_01)
net:add(nn.Tanh())
net:add(module_02)

-- Set weights and biases
--net:get(1).bias[1] = 12
--net:get(1).weight[1] = 2

-- For back propagation
criterion = nn.MSECriterion(1)
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = learningRate
trainer.maxIteration = 1



local startTime = os.clock()

for epoch = 1, EPOCH_TIMES do
    print(string.format('Epoch %d', epoch))

    for m = 1, NUM_BATCHES do
        print("Batch %d of %d", m, NUM_BATCHES)
        trainer:train(train)
    end
end


print("Duration: " .. os.clock() - startTime .. "s")

-- Plot
gnuplot.pngfigure('graph/firstCombination.png')
gnuplot.title('First Combination - Morning Values')
gnuplot.xlabel('Day')
gnuplot.ylabel('Glucose Level')
gnuplot.plot({'Prediction', prediction}, {'Expectation', expectation})
gnuplot.plotflush()
