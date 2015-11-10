--[[
-- Author: Hai Tran
-- Date: Nov 09, 2015
-- Filename: criterion.lua
-- Descriptiion: This file is a practice on criterion.
--]]
require 'nn'
require '../bgl_dataLoading.lua'

function gradientUpgrade(model, x, y, criterion, learningRate)
	local prediction = model:forward(x)
	local err = criterion:forward(prediction, y)
	local gradOutputs = criterion:backward(prediction, y)
	model:zeroGradParameters()
	model:backward(x, gradOutputs)
	model:updateParameters(learningRate)
end

local path = "../../Datasets/Peter Kok - Real data for predicting blood glucose levels of diabetics/data.txt"
local date, time, glucose, shortActingInsulin, longActingInsulin, food, exercise, stress = loadFile(path)

local NUM_OF_INPUTS = 4
local NUM_OF_OUTPUTS = 1

net = nn.Sequential()
net:add(nn.Linear(NUM_OF_INPUTS, NUM_OF_OUTPUTS))

-- Mean Square Error Criterion
criterion = nn.MSECriterion()

-- for i = 1, 30 do
i = 3
input = torch.Tensor({glucose[4*i], shortActingInsulin[4*i], food[4*i], exercise[4*i]})
output = torch.Tensor({glucose[4*i+1]})
-- end

print('glucose = ' .. glucose[4*i])
print('short acting insulin = ' .. shortActingInsulin[4*i])
print('long acting insulin = ' .. longActingInsulin[4*i])
print('exercise = ' .. exercise[4*i])

for j = 1, 1000 do
	gradientUpgrade(net, input, output, criterion, 0.01)
end

print('prediction for input = ' .. net:forward(input)[1] .. ' expected value ' .. output[1])