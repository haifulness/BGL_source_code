--[[
-- Author: Hai Tran
-- Date: Jan 06, 2016
-- Filename: bgl_all_intervals_regularization.lua
]]

require 'torch'
require 'nn'
require 'gnuplot'
require 'optim'
require("bgl_dataLoading.lua")
require("bgl_generateSets.lua")


local SIZE_INPUT = 10
local SIZE_HIDDEN_LAYER = 100
local SIZE_OUTPUT = 1

local ACCEPT_THRESHOLD = 1e-5
local LAMBDA = 1e-5
local EPOCH_TIMES = 2*1e5
local learningRate = 1e-5
local epoch = 1
local threshold = 1
local minThreshold = 1
local bestError = 0

local net = nn.Sequential()
criterion = nn.MSECriterion(true)
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



-- parse command-line options
--
local opt = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
   -f,--full                                use the full dataset
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 10)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
]]
opt.learningRate = learningRate
opt.batchSize = 10
opt.momentum = 0
opt.maxIter = 3
opt.coefL1 = LAMBDA
opt.coefL2 = LAMBDA


-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)


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

local train, test, validation = {}, {}, {}
local trainErr, validationErr, testErr = {}, {}, {}
local train_input  = torch.Tensor(#train, SIZE_INPUT)
local train_output = torch.Tensor(#train)
local test_input  = torch.Tensor(#test, SIZE_INPUT)
local test_output = torch.Tensor(#test)
local validation_input  = torch.Tensor(#validation, SIZE_INPUT)
local validation_output = torch.Tensor(#validation)

-- retrieve parameters and gradients
parameters, gradParameters = net:getParameters()

function randomSets()
    -- Divide the dataset
    train, test, validation = generateSets(NUM_DAY * 4 - 4, 70, 20, 10)
    -- Input and output of the neural net
    train_input  = torch.Tensor(#train, SIZE_INPUT)
    train_output = torch.Tensor(#train)
    test_input  = torch.Tensor(#test, SIZE_INPUT)
    test_output = torch.Tensor(#test)
    validation_input  = torch.Tensor(#validation, SIZE_INPUT)
    validation_output = torch.Tensor(#validation)

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
end



-- train
function trainNet()
    -- epoch tracker
    if epoch < 1 then epoch = 1 end

    -- do one epoch
    print("\nEpoch # " .. epoch .. '')

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
        -- just in case:
        collectgarbage()

        -- get new parameters
        if x ~= parameters then
            parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        -- evaluate function for complete mini batch
        local outputs = net:forward(train_input)
        local f = criterion:forward(outputs, train_output)
        
        print('Training error = ' .. f)
        trainErr[epoch] = f

        -- estimate df/dW
        local df_do = criterion:backward(outputs, train_output)
        net:backward(train_input, df_do)

        -- penalties (L1 and L2):
        if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm, sign = torch.norm, torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters, 1)
            f = f + opt.coefL2 * norm(parameters, 2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
        end

        -- return f and df/dX
        return f, gradParameters
    end


    -- Perform SGD step:
    sgdState = sgdState or {
        learningRate = opt.learningRate,
        momentum = opt.momentum,
        learningRateDecay = 5e-7
    }
    optim.sgd(feval, parameters, sgdState)
end



-- test
function testNet()
    local prediction = net:forward(test_input)
    testErr[epoch] = criterion:forward(prediction, test_output)
    print("Test Error = " .. testErr[epoch])
end

--------------------------
-- Main

-- timer
local startTime = sys.clock()

while threshold > ACCEPT_THRESHOLD and epoch < EPOCH_TIMES do
    randomSets()
    trainNet()
    
    local prediction = net:forward(validation_input)
    validationErr[epoch] = criterion:forward(prediction, validation_output)
    print('Validation error = ' .. validationErr[epoch])

    testNet()
    threshold = math.abs(trainErr[epoch] - testErr[epoch])
    if threshold < minThreshold then
        minThreshold = threshold
        bestError = testErr[epoch]
    end

    epoch = epoch + 1
end

print('\nTraining Time: ' .. sys.clock() - startTime)
print('Best Error: ' .. bestError)

-- Plot

-- only plot once for every 1000 epoches
local train_selected, validation_selected, test_selected = {}, {}, {}

for i = 1, EPOCH_TIMES do
    if i % 1e3 == 0 then
        train_selected[math.ceil(i/1e3)] = trainErr[i]
        validation_selected[math.ceil(i/1e3)] = validationErr[i]
        test_selected[math.ceil(i/1e3)] = testErr[i]
    end
end

gnuplot.pngfigure('graph/Jan 13/all_intervals_regularization/error.png')
gnuplot.title('All Intervals - Error')
gnuplot.ylabel('Glucose Level')
gnuplot.xlabel('Epoch (x1000)')
gnuplot.plot(
    {'Train Error', torch.Tensor(train_selected)}, 
    {'Validation Error', torch.Tensor(validation_selected)}, 
    {'Test Error', torch.Tensor(test_selected)})
gnuplot.plotflush()


--[[
--
-- SAVE
--
--]]
torch.save("graph/Jan 13/all_intervals_regularization/0.model", net)