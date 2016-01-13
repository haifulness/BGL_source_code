--[[
-- Author: Hai Tran
-- Date: Dec 24, 2015
-- Filename: bgl_all_intervals.lua
]]

require 'torch'
require 'nn'
require 'gnuplot'
require("bgl_dataLoading.lua")
require("bgl_generateSets.lua")

-- Neural net model:
--   + Linear
--   +  inputs
--   + Multiple hidden layer with 100 nodes each
--   + 1 output
--
local SIZE_INPUT = 10
local SIZE_HIDDEN_LAYER = 100
local SIZE_OUTPUT = 1

local EPOCH_TIMES = 1e5
local threshold = 0.001        -- For validation set
local learningRate = 1e-5
local thresholdMet = false
local epoch = 0
local minError = 100         -- The error should be lower than this value
local sumError = 0

local net = nn.Sequential()
criterion = nn.MSECriterion(1)
module_01 = nn.Linear(SIZE_INPUT, SIZE_HIDDEN_LAYER)
module_02 = nn.Linear(SIZE_HIDDEN_LAYER, SIZE_HIDDEN_LAYER)
module_03 = nn.Linear(SIZE_HIDDEN_LAYER, SIZE_HIDDEN_LAYER)
module_04 = nn.Linear(SIZE_HIDDEN_LAYER, SIZE_HIDDEN_LAYER)
module_05 = nn.Linear(SIZE_HIDDEN_LAYER, SIZE_HIDDEN_LAYER)
module_06 = nn.Linear(SIZE_HIDDEN_LAYER, SIZE_OUTPUT)
net:add(module_01)
--net:add(module_02)
--net:add(module_03)
--net:add(module_04)
--net:add(module_05)
net:add(nn.Tanh())
net:add(module_06)

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



-- Build the data table
local input = {}
local output = {}
for i = 1, NUM_DAY * 4 - 4 do
    input[i] = {}

    -- morning
    if i % 4 == 1 then
        local k = math.floor(i/4) + 1

        -- BGL of the most recent interval
        input[i][1]  = night_glucose[k]

        -- Long acting insulin shots in the last 24 hours
        input[i][2]  = morning_LAI[k]
        input[i][3]  = afternoon_LAI[k]
        input[i][4]  = evening_LAI[k]
        input[i][5]  = night_LAI[k]

        -- Everything else in the same interval
        input[i][6]  = morning_SAI[k + 1]
        input[i][7]  = morning_LAI[k + 1]
        input[i][8]  = morning_food[k + 1]
        input[i][9]  = morning_exercise[k + 1]
        input[i][10] = morning_stress[k + 1]

        -- Output
        output[i]    = morning_glucose[k + 1]

    end

    -- afternoon
    if i % 4 == 2 then
        local k = math.floor(i/4) + 1

        -- BGL of the most recent interval
        input[i][1]  = morning_glucose[k + 1]

        -- Long acting insulin shots in the last 24 hours
        input[i][2]  = afternoon_LAI[k]
        input[i][3]  = evening_LAI[k]
        input[i][4]  = night_LAI[k]
        input[i][5]  = morning_LAI[k + 1]
        
        -- Everything else in the same interval
        input[i][6]  = afternoon_SAI[k + 1]
        input[i][7]  = afternoon_LAI[k + 1]
        input[i][8]  = afternoon_food[k + 1]
        input[i][9]  = afternoon_exercise[k + 1]
        input[i][10] = afternoon_stress[k + 1]

        -- Output
        output[i]    = afternoon_glucose[k + 1]
        
    end

    -- evening
    if i % 4 == 3 then
        local k = math.floor(i/4) + 1

        -- BGL of the most recent interval
        input[i][1]  = afternoon_glucose[k + 1]

        -- Long acting insulin shots in the last 24 hours
        input[i][2]  = evening_LAI[k]
        input[i][3]  = night_LAI[k]
        input[i][4]  = morning_LAI[k + 1]
        input[i][5]  = afternoon_LAI[k + 1]

        -- Everything else in the same interval
        input[i][6]  = evening_SAI[k + 1]
        input[i][7]  = evening_LAI[k + 1]
        input[i][8]  = evening_food[k + 1]
        input[i][9]  = evening_exercise[k + 1]
        input[i][10] = evening_stress[k + 1]

        -- Output
        output[i]    = evening_glucose[k + 1]

    end

    -- night
    if i % 4 == 0 then
        local k = math.floor(i/4)

        -- BGL of the most recent interval
        input[i][1]  = evening_glucose[k + 1]

        -- Long acting insulin shots in the last 24 hours
        input[i][2]  = night_LAI[k]
        input[i][3]  = morning_LAI[k + 1]
        input[i][4]  = afternoon_LAI[k + 1]
        input[i][5]  = evening_LAI[k + 1]

        -- Everything else in the same interval
        input[i][6]  = night_SAI[k + 1]
        input[i][7]  = night_LAI[k + 1]
        input[i][8]  = night_food[k + 1]
        input[i][9]  = night_exercise[k + 1]
        input[i][10] = night_stress[k + 1]

        -- Output
        output[i]    = night_glucose[k + 1]

    end
end


-- Divide the dataset
local train, test, validation = generateSets(NUM_DAY * 4 - 4, 50, 20, 30)
-- Input and output of the neural net
local train_input  = torch.Tensor(#train, SIZE_INPUT)
local train_output = torch.Tensor(#train)
local test_input  = torch.Tensor(#test, SIZE_INPUT)
local test_output = torch.Tensor(#test)
local validation_input  = torch.Tensor(#validation, SIZE_INPUT)
local validation_output = torch.Tensor(#validation)

input_storage, output_storage, counter = {}, {}, 0
-- Load data into input & output
for key, val in pairs(train) do
    counter = counter + 1
    input_storage[counter] = {}

    input_storage[counter][1]  = input[val][1]
    input_storage[counter][2]  = input[val][2]
    input_storage[counter][3]  = input[val][3]
    input_storage[counter][4]  = input[val][4]
    input_storage[counter][5]  = input[val][5]
    input_storage[counter][6]  = input[val][6]
    input_storage[counter][7]  = input[val][7]
    input_storage[counter][8]  = input[val][8]
    input_storage[counter][9]  = input[val][9]
    input_storage[counter][10] = input[val][10]

    output_storage[counter] = output[val]
end

-- Convert input to a Tensor
train_input  = torch.Tensor(input_storage)
train_output = torch.Tensor(output_storage)


input_storage, output_storage, counter = {}, {}, 0
-- Load data into input & output
for key, val in pairs(test) do
    counter = counter + 1
    input_storage[counter] = {}

    input_storage[counter][1]  = input[val][1]
    input_storage[counter][2]  = input[val][2]
    input_storage[counter][3]  = input[val][3]
    input_storage[counter][4]  = input[val][4]
    input_storage[counter][5]  = input[val][5]
    input_storage[counter][6]  = input[val][6]
    input_storage[counter][7]  = input[val][7]
    input_storage[counter][8]  = input[val][8]
    input_storage[counter][9]  = input[val][9]
    input_storage[counter][10] = input[val][10]

    output_storage[counter] = output[val] 
end

-- Convert input to a Tensor
test_input  = torch.Tensor(input_storage)
test_output = torch.Tensor(output_storage)

input_storage, output_storage, counter = {}, {}, 0
-- Load data into input & output
for key, val in pairs(validation) do
    counter = counter + 1
    input_storage[counter] = {}

    input_storage[counter][1]  = input[val][1]
    input_storage[counter][2]  = input[val][2]
    input_storage[counter][3]  = input[val][3]
    input_storage[counter][4]  = input[val][4]
    input_storage[counter][5]  = input[val][5]
    input_storage[counter][6]  = input[val][6]
    input_storage[counter][7]  = input[val][7]
    input_storage[counter][8]  = input[val][8]
    input_storage[counter][9]  = input[val][9]
    input_storage[counter][10] = input[val][10]

    output_storage[counter] = output[val] 
end

-- Convert input to a Tensor
validation_input  = torch.Tensor(input_storage)
validation_output = torch.Tensor(output_storage)



local startTime = os.clock()
local prediction
local trainErr = {}
local validationErr = {}
local testErr = {}


-- Train
for i = 1, EPOCH_TIMES do
    gradientUpgrade(net, train_input, train_output, criterion, learningRate)
    prediction = net:forward(test_input)
    trainErr[i] = criterion:forward(prediction, test_output)
end

-- Validate
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
--]]

--[[ 
--
-- TEST
--
]]

prediction = net:forward(test_input)
local err = criterion:forward(prediction, test_output)

print("\nTraining Duration: " .. os.clock() - startTime .. "")
print("Smallest Validation Error: " .. minError)
print("Average Validation Error: " .. sumError / EPOCH_TIMES)
print("Test Error: " .. err)

-- Plot
gnuplot.pngfigure('graph/Dec 25/all_intervals.png')
gnuplot.title('All Intervals')
gnuplot.ylabel('Glucose Level')
gnuplot.plot({'Prediction', prediction}, {'Expectation', test_output})
gnuplot.plotflush()

gnuplot.pngfigure('graph/Dec 25/all_intervals_error.png')
gnuplot.title('All Intervals - Error')
gnuplot.ylabel('Glucose Level')
gnuplot.plot({'Train Error', torch.Tensor(trainErr)}, 
    {'Validation Error', torch.Tensor(validationErr)})
gnuplot.plotflush()
--]]


--[[
--
-- SAVE
--
--]]
torch.save("model", net)
