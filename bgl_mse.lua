--[[
-- Author: Hai Tran
-- Date: 12/02/2015
-- Filename: bgl_mse.lua
--
-- Calculate the mean squared error between two sets
]]

function mse(expectation, prediction)
    local sum = 0
    
    for i = 1, #expectation do
        sum = sum + math.pow(expectation[i] - prediction[i], 2)
    end

    return (sum / #expectation)
end