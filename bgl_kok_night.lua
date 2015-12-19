--[[
-- Author: Hai Tran
-- Date: Dec 13, 2015
-- Filename: bgl_kok_night.lua
-- Descriptiion: Simulate Peter Kok's suggestion on the combination that best
-- simulates the glucose level in the evening.
-- This combination consists of:
--   + Glucose level (during the interval)
--   + Short acting insulin (during the interval)
--   + Food intake (during the interval)
--   + Exercise (during the interval)
--   + Glucose level (previous interval)
--   + Short acting insulin (previous interval)
--   + Food intake (previous interval)
--   + Exercise (previous interval)
]]

require 'torch'
require 'nn'
require 'gnuplot'
require("bgl_dataLoading.lua")
require("bgl_generateSets.lua")

-- Neural net model:
--   + Linear
--   + 8 inputs
--   + 1 hidden layer with 100 nodes
--   + 1 output
--
local SIZE_INPUT = 8
local SIZE_HIDDEN_LAYER = 100
local SIZE_OUTPUT = 1

local EPOCH_TIMES = 100000
local threshold = 0.00001    -- For validation set
local learningRate = 0.0001
local thresholdMet = false
local epoch = 0
local minError = 100         -- The error should be lower than this value
local sumError = 0

local net = nn.Sequential()
criterion = nn.MSECriterion(1)
module_01 = nn.Linear(SIZE_INPUT, SIZE_HIDDEN_LAYER)
module_02 = nn.Linear(SIZE_HIDDEN_LAYER, SIZE_OUTPUT)
net:add(module_01)
net:add(nn.Sigmoid())
net:add(module_02)
-- Set weights and biases
--net:get(1).bias[1] = 2
--net:get(1).weight[1] = 2

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
end



-- Divide the dataset
local train, test, validation = generateSets(NUM_DAY - 1, 40, 30, 30)

-- Input and output of the neural net
local train_input  = torch.Tensor(#train, SIZE_INPUT)
local train_output = torch.Tensor(#train)
local test_input  = torch.Tensor(#test, SIZE_INPUT)
local test_output = torch.Tensor(#test)
local validation_input  = torch.Tensor(#validation, SIZE_INPUT)
local validation_output = torch.Tensor(#validation)

local input_storage, output_storage, counter = {}, {}, 0
-- Load data into input & output
for key, val in pairs(train) do
    counter = counter + 1
    input_storage[counter] = {}

    input_storage[counter][1] = evening_glucose[val]
    input_storage[counter][2] = night_SAI[val]
    input_storage[counter][3] = night_food[val]
    input_storage[counter][4] = night_exercise[val]
    input_storage[counter][5] = night_stress[val]
    input_storage[counter][6] = night_LAI[val - 1]
    input_storage[counter][7] = night_exercise[val - 1] * night_exercise[val - 1]
    input_storage[counter][8] = 6  -- assume the interval length is 6 hours
    output_storage[counter]   = night_glucose[val]
end
-- Convert input to a Tensor
train_input  = torch.Tensor(input_storage)
train_output = torch.Tensor(output_storage)


input_storage, output_storage, counter = {}, {}, 0
-- Load data into input & output
for key, val in pairs(test) do
    counter = counter + 1
    input_storage[counter] = {}

    input_storage[counter][1] = evening_glucose[val]
    input_storage[counter][2] = night_SAI[val]
    input_storage[counter][3] = night_food[val]
    input_storage[counter][4] = night_exercise[val]
    input_storage[counter][5] = night_stress[val]
    input_storage[counter][6] = night_LAI[val - 1]
    input_storage[counter][7] = night_exercise[val - 1] * night_exercise[val - 1]
    input_storage[counter][8] = 6  -- assume the interval length is 6 hours
    output_storage[counter]   = night_glucose[val]
end
-- Convert input to a Tensor
test_input  = torch.Tensor(input_storage)
test_output = torch.Tensor(output_storage)


input_storage, output_storage, counter = {}, {}, 0
-- Load data into input & output
for key, val in pairs(validation) do
    counter = counter + 1
    input_storage[counter] = {}

    input_storage[counter][1] = evening_glucose[val]
    input_storage[counter][2] = night_SAI[val]
    input_storage[counter][3] = night_food[val]
    input_storage[counter][4] = night_exercise[val]
    input_storage[counter][5] = night_stress[val]
    input_storage[counter][6] = night_LAI[val - 1]
    input_storage[counter][7] = night_exercise[val - 1] * night_exercise[val - 1]
    input_storage[counter][8] = 6  -- assume the interval length is 6 hours
    output_storage[counter]   = night_glucose[val]
end
-- Convert input to a Tensor
validation_input  = torch.Tensor(input_storage)
validation_output = torch.Tensor(output_storage)




local startTime = os.clock()
local prediction
local trainErr = {}
local validationErr = {}
local testErr = {}

for i = 1, EPOCH_TIMES do
    -- Train
    gradientUpgrade(net, train_input, train_output, criterion, learningRate)
    prediction = net:forward(test_input)
    trainErr[i] = criterion:forward(prediction, test_output)
end

while not thresholdMet and epoch < EPOCH_TIMES do
    epoch = epoch + 1
    print('\nEpoch #' .. epoch)

    -- Train
    gradientUpgrade(net, train_input, train_output, criterion, learningRate)
 
    -- Validate
    prediction = net:forward(validation_input)
    validationErr[epoch] = criterion:forward(prediction, validation_output)
    print("Validation Error: " .. validationErr[epoch])
    if validationErr[epoch] < threshold then thresholdMet = true end 
    if validationErr[epoch] < minError then minError = validationErr[epoch] end
    sumError = sumError + validationErr[epoch]
end


--[[ 
--
-- TEST
--
]]
--for i = 1, #test do
    prediction = net:forward(test_input)
    local err = criterion:forward(prediction, test_output)
--end
print("\nTraining Duration: " .. os.clock() - startTime .. "s")
print("Smallest Validation Error: " .. minError)
print("Average Validation Error: " .. sumError / EPOCH_TIMES)
print("Test Error: " .. err)

-- Plot
gnuplot.pngfigure('graph/Dec 13/kok_night.png')
gnuplot.title('Peter Kok\'s Choice For Night')
gnuplot.ylabel('Glucose Level')
gnuplot.plot({'Prediction', prediction}, {'Expectation', test_output})
gnuplot.plotflush()


gnuplot.pngfigure('graph/Dec 13/kok_night_error.png')
gnuplot.title('Peter Kok\'s Choice For Night - Error')
gnuplot.ylabel('Glucose Level')
gnuplot.plot({'Train Error', torch.Tensor(trainErr)}, 
    {'Validation Error', torch.Tensor(validationErr)})
gnuplot.plotflush()



--[[
--
-- Save the net & the datasets to files
--
]]
local file = io.open("datasets.txt", "w")

file:write(tostring(#train))
file:write(";")
file:write(tostring(#test))
file:write(";")
file:write(tostring(#validation))
file:write(";")

for key, val in pairs(train) do
    file:write(tostring(val))
    file:write(";")
end
for key, val in pairs(test) do
    file:write(tostring(val))
    file:write(";")
end
for key, val in pairs(validation) do
    file:write(tostring(val))
    file:write(";")
end
file:close()


file = io.open("net_night.txt", "w")
for i = 1, SIZE_HIDDEN_LAYER do
    for j = 1, SIZE_INPUT do
        local w = net:get(1).weight[i][j]
        file:write(tostring(w))
        file:write(";")
    end
end
file:close()

