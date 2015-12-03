--[[
-- Author: Hai Tran
-- Date: Nov 29, 2015
-- Filename: bgl_first.lua
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
require("bgl_activationFunctions.lua")
require("bgl_feedForward.lua")
require("bgl_mse.lua")

-- Seed the random function
math.randomseed(os.time())

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
local NUM_DAY = #morning_date

-- Percentage size of the train, test, and validation sets
local PERCENT_TRAIN, PERCENT_TEST, PERCENT_VALIDATION = 60, 20, 20

-- Neural net model:
--   4 inputs
--   1 hidden layer with 10 nodes
--   1 output
local SIZE_INPUT = 4
local SIZE_HIDDEN_LAYER = 10
local SIZE_OUTPUT = 1

-- Define the activation function for the net. We can replace the inner
-- function by another one.
function activationFunction(x)
    return sigmoid(x)
end

local input, prediction, expectation = {}, {}, {}

-- Build the input
for i = 1, NUM_DAY - 1 do
    input[i]       = {}
    input[i][1]    = morning_glucose[i]
    input[i][2]    = morning_SAI[i]
    input[i][3]    = morning_food[i]
    input[i][4]    = morning_exercise[i]
    expectation[i] = morning_glucose[i + 1]
end

-- Bias
local bias = math.random()

-- Random the weights
local weight = {}
for i = 1, SIZE_OUTPUT do
    weight[i] = {}
    for j = 1, SIZE_INPUT + 1 do
        weight[i][j] = math.random()
    end
end

prediction[1] = feedForward(input[1], weight, bias)
--print(mse(expectation[1], prediction[1][1]))
local error = expectation[1] - prediction[1][1]
print('Expectation: ' .. expectation[1])
print('Prediction: ' .. prediction[1][1])
print('Error: ' .. error)

-- Divide the dataset
local train, test, validation 
    = generateSets(NUM_DAY - 1, PERCENT_TRAIN, PERCENT_TEST, PERCENT_VALIDATION)

-- Plot
--[[
gnuplot.setterm('x11')
gnuplot.pngfigure('graph/firstCombination.png')
gnuplot.title('First Combination - Morning Values')
gnuplot.xlabel('Day')
gnuplot.ylabel('Glucose Level')
gnuplot.plot({'Prediction', output}, {'Expectation', expectation})
gnuplot.plotflush()
]]--