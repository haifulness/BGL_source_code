--[[
-- Author: Hai Tran
-- Date: Jan 17, 2016
-- Filename: bgl_kok_evening_regularization.lua
]]

require 'torch'
require 'nn'
require 'gnuplot'
require 'optim'
require("bgl_dataLoading.lua")
require("bgl_generateSets.lua")


local SIZE_INPUT = 8
local SIZE_HIDDEN_LAYER = 100
local SIZE_OUTPUT = 1

local ACCEPT_THRESHOLD = 1e-5
local LAMBDA = 1e-5
local EPOCH_TIMES = 1*1e6
local learningRate = 1e-5
local epoch = 1
local threshold = 1
local minThreshold = 100

local net = nn.Sequential()
criterion = nn.MSECriterion(true)
module_01 = nn.Linear(SIZE_INPUT, SIZE_HIDDEN_LAYER)
module_02 = nn.Linear(SIZE_HIDDEN_LAYER, SIZE_HIDDEN_LAYER)
module_03 = nn.Linear(SIZE_HIDDEN_LAYER, SIZE_HIDDEN_LAYER)
module_04 = nn.Linear(SIZE_HIDDEN_LAYER, SIZE_HIDDEN_LAYER)
module_05 = nn.Linear(SIZE_HIDDEN_LAYER, SIZE_HIDDEN_LAYER)
module_06 = nn.Linear(SIZE_HIDDEN_LAYER, SIZE_OUTPUT)
net:add(module_01)
net:add(module_02)
net:add(module_03)
net:add(module_04)
net:add(module_05)
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
    train, test, validation = generateSets(NUM_DAY - 1, 70, 20, 10)

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

        input_storage[counter][1] = afternoon_glucose[val]
        input_storage[counter][2] = evening_SAI[val]
        input_storage[counter][3] = evening_food[val]
        input_storage[counter][4] = evening_exercise[val]
        input_storage[counter][5] = morning_glucose[val]
        input_storage[counter][6] = afternoon_SAI[val]
        input_storage[counter][7] = afternoon_food[val]
        input_storage[counter][8] = afternoon_exercise[val]
        output_storage[counter]   = evening_glucose[val]
    end
    -- Convert input to a Tensor
    train_input  = torch.Tensor(input_storage)
    train_output = torch.Tensor(output_storage)


    input_storage, output_storage, counter = {}, {}, 0
    -- Load data into input & output
    for key, val in pairs(test) do
        counter = counter + 1
        input_storage[counter] = {}

        input_storage[counter][1] = afternoon_glucose[val]
        input_storage[counter][2] = evening_SAI[val]
        input_storage[counter][3] = evening_food[val]
        input_storage[counter][4] = evening_exercise[val]
        input_storage[counter][5] = morning_glucose[val]
        input_storage[counter][6] = afternoon_SAI[val]
        input_storage[counter][7] = afternoon_food[val]
        input_storage[counter][8] = afternoon_exercise[val]
        output_storage[counter]   = evening_glucose[val]
    end
    -- Convert input to a Tensor
    test_input  = torch.Tensor(input_storage)
    test_output = torch.Tensor(output_storage)


    input_storage, output_storage, counter = {}, {}, 0
    -- Load data into input & output
    for key, val in pairs(validation) do
        counter = counter + 1
        input_storage[counter] = {}

        input_storage[counter][1] = afternoon_glucose[val]
        input_storage[counter][2] = evening_SAI[val]
        input_storage[counter][3] = evening_food[val]
        input_storage[counter][4] = evening_exercise[val]
        input_storage[counter][5] = morning_glucose[val]
        input_storage[counter][6] = afternoon_SAI[val]
        input_storage[counter][7] = afternoon_food[val]
        input_storage[counter][8] = afternoon_exercise[val]
        output_storage[counter]   = evening_glucose[val]
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

local pathPrefix = 'graph/Jan 25/evening/'
local bestError, duration = {}, {}
local resultFile = pathPrefix .. 'result.txt'

for index = 1, 10 do
    local file, fileErr = io.open(resultFile, 'a+')

    if fileErr then print('File Open Error')
    else
        bestError[index] = 100
        duration[index] = 0

        -- reset
        epoch = 1
        threshold = 1
        minThreshold = 100

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
                bestError[index] = testErr[epoch]
            end

            epoch = epoch + 1
        end

        duration[index] = sys.clock() - startTime
        print('\nTraining Time: ' .. duration[index])
        print('Best Test Error: ' .. bestError[index])


        ----- Plot -----

        -- only plot once for every 1000 epoches
        local train_selected, validation_selected, test_selected = {}, {}, {}

        for i = 1, EPOCH_TIMES do
            if i % 1e4 == 0 then
                train_selected[math.ceil(i/1e4)] = trainErr[i]
                validation_selected[math.ceil(i/1e4)] = validationErr[i]
                test_selected[math.ceil(i/1e4)] = testErr[i]
            end
        end

        
        local graphFile = pathPrefix .. 'error_' .. index .. '.png'
        gnuplot.pngfigure(graphFile)
        gnuplot.title('Evening Interval - Error')
        gnuplot.ylabel('Glucose Level')
        gnuplot.xlabel('Epoch (x1000)')
        gnuplot.plot(
            {'Train Error', torch.Tensor(train_selected)}, 
            --{'Validation Error', torch.Tensor(validation_selected)}, 
            {'Test Error', torch.Tensor(test_selected)})
        gnuplot.plotflush()


        --[[
        --
        -- SAVE
        --
        --]]
        torch.save(pathPrefix .. '' .. 'model_' .. index, net)
        file:write('\n----- Run #' .. index .. ' -----\n')
        file:write('Duration: ' .. duration[index] .. '\n')
        file:write('Best Test Error: ' .. bestError[index] .. '\n')

    end

    file:close()
end
