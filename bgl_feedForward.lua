--[[
-- Author: Hai Tran
-- Date: 12/02/2015
-- Filename: bgl_feedForward.lua
--
-- Define the feed forward function of a neural network
--]]

function feedForward(input, weight, bias)
    local output = {}

    for i = 1, #weight do
    	output[i] = bias * weight[i][#input + 1]

    	for j = 1, #input do
	        output[i] = output[i] + weight[i][j] * input[j]
	    end
    end

    return output
end
