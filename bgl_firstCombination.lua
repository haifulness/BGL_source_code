--[[
-- Author: Hai Tran
-- Date: Nov 21, 2015
-- Filename: bgl_firstCombination.lua
-- Descriptiion: Simulate the first combination that Peter Kok suggested.
]]

require 'torch'
require 'nn'
require 'gnuplot'
require("bgl_dataLoading2.lua")
require("bgl_generateSets.lua")

gnuplot.setterm('x11')

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


function calculate(model, input)
    local ret = model:forward(input)
    return ret
end

function gradientUpgrade(model, input, output, criterion, learningRate)
    local prediction  = model:forward(input)
    local err         = criterion:forward(prediction, output)
    local gradOutputs = criterion:backward(prediction, output)

    model:zeroGradParameters()
    model:backward(input, gradOutputs)
    model:updateParameters(learningRate)
end

--[[
-- The first combination consists of:
--   + Glucose level (at the start of the interval)
--   + Short acting insulin (during the interval)
--   + Food intake (during the interval)
--   + Exercise (during the interval)
]]
NUM_DAY = #morning_date
local train, test, validate = {}, {}, {}

-- Neural net model
-- 4 inputs
-- 1 hidden layer with 10 nodes
-- 1 output
local SIZE_INPUT = 4
local SIZE_HIDDEN_LAYER = 10
local SIZE_OUTPUT = 1
local net = nn.Sequential()
--net:add(nn.Linear(SIZE_INPUT, SIZE_HIDDEN_LAYER))
--net:add(nn.Linear(SIZE_HIDDEN_LAYER, SIZE_OUTPUT))
local module_01 = nn.Linear(SIZE_INPUT, SIZE_OUTPUT)
net:add(module_01)

criterion = nn.MarginCriterion(1)

-- Divide the dataset
train, test, validate = generateSets(NUM_DAY - 1, 60, 20, 20)
local input  = torch.Tensor(NUM_DAY, SIZE_INPUT)
local output = torch.Tensor(NUM_DAY, SIZE_OUTPUT)

for i = 1, NUM_DAY - 1 do
    -- Build the input
    -- I'll try with the morning set first
    local input_storage = torch.Storage(4)
    input_storage[1]    = morning_glucose[i]
    input_storage[2]    = morning_SAI[i]
    input_storage[3]    = morning_food[i]
    input_storage[4]    = morning_exercise[i]

    -- Convert input to a Tensor
    input[i]  = torch.Tensor(input_storage)
    --local output = torch.Tensor({1})

    -- Train
    output[i] = calculate(net, input[i])
    --gradientUpgrade(net, input, output, criterion, 0.01)

    print('\nDay #' .. i)
    print('Prediction: ' .. output[i][1])
    print('Expectation: ' .. morning_glucose[i+1])

    -- Test

    -- Validate
end

-- Create a tensor for expectation values
local expectation_storage = torch.Storage(NUM_DAY-1)
for i = 1, NUM_DAY-1 do
    expectation_storage[i] = morning_glucose[i+1]
end
local expectation = torch.Tensor(expectation_storage)

-- Plot
gnuplot.pngfigure('firstCombination.png')
gnuplot.title('First Combination')
gnuplot.xlabel('Day')
gnuplot.ylabel('Glucose Level')
gnuplot.plot(
    {'Prediction', output}, 
    {'Expectation', expectation})
gnuplot.plotflush()
