--[[
-- Author: Hai Tran
-- Date: Jan 02, 2016
-- Filename: bgl_rnn.lua
-- Recurrent Neural Network
]]

require 'torch'
require 'nn'
require 'rnn'
require 'gnuplot'
require("bgl_dataLoading.lua")
require("bgl_generateSets.lua")


local SIZE_INPUT = 10
local SIZE_HIDDEN_LAYER = 100
local SIZE_OUTPUT = 1

local EPOCH_TIMES = 1e5
local learningRate = 1e-2
local epoch = 0
local minError = 100         
local sumError = 0
local rho = 5                -- maximum number of steps to backpropagate through time (BPTT)
local updateInterval = 4

-- Divide the dataset
NUM_DAY = 77
local train, test, validation = generateSets(NUM_DAY * 4 - 4, 100, 00, 00)
local trainErr = {}
criterion = nn.MSECriterion()

local rnn = nn.Recurrent(
        SIZE_HIDDEN_LAYER,
        nn.Linear(SIZE_INPUT, SIZE_HIDDEN_LAYER),
        nn.Linear(SIZE_HIDDEN_LAYER, SIZE_HIDDEN_LAYER),
        nn.Sigmoid(), -- transfer function
        rho -- maximum number of time-steps for BPTT
    )
local net = nn.Sequential()
net:add(rnn)
net:add(nn.Linear(SIZE_HIDDEN_LAYER, SIZE_OUTPUT))


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


function gradientUpgrade(model, x, y, criterion, learningRate, i)
    local prediction = model:forward(x)
    local err = criterion:forward(prediction, y)
    local gradOutputs = criterion:backward(prediction, y)
    -- the Recurrent layer is memorizing its gradOutputs (up to memSize)
    model:backward(x, gradOutputs)

    --if i % 100 == 0 then
    print('error for iteration ' .. i  .. ' is ' .. err/rho)
    table.insert(trainErr, err/rho)
    if (err/rho < minError) then minError = err/rho end
    sumError = sumError + err/rho
    --end

    if i % updateInterval == 0 then
        -- backpropagates through time (BPTT) :
        -- 1. backward through feedback and input layers,
        -- 2. updates parameters
        model:backwardThroughTime()
        model:updateParameters(learningRate)
        model:zeroGradParameters()
    end
end


--[[
--
-- TRAINING
--
--]]
local startTime = os.clock()
for epoch = 1, EPOCH_TIMES do
    gradientUpgrade(net, train_input, train_output, criterion, learningRate, epoch)
end

print('\nTraining Duration: ' .. os.clock() - startTime .. '')
print('Smallest Error: ' .. minError .. '\nAverage Error: ' .. sumError/EPOCH_TIMES)

--[[
-- Plot
--]]
gnuplot.pngfigure('graph/Jan 05/_error.png')
gnuplot.title('Recurrent Neural Network')
gnuplot.ylabel('Error')
gnuplot.plot({'Train Error', torch.Tensor(trainErr)})
gnuplot.plotflush()


--[[
--
-- SAVE
--
--]]
torch.save("graph/Jan 05/model", net)