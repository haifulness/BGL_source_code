--[[
-- Author: Hai Tran
-- Date: 12/02/2015
-- Filename: bgl_activationFunction.lua
--
-- A collection of mathematical functions to be used as activation function
-- for a neural network.
]]

-- Sigmoid function
function sigmoid(x)
    return (1.0 / math.exp(-x))
end

function sigmoid_derivate(x)
    return x * (1 - x)
end


-- Since Lua already has a built-in math.tanh() function,
-- we only need to implement the derivative version of it.
function tanh_derivative(x)
    return (1 - math.pow(math.tanh(x), 2)) * 0.5
end