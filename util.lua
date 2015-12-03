--[[
-- Author: Hai Tran
-- Date: Nov 28, 2015
-- File name: bgl_util.lua
--
-- Provides necessary functionalities for the neural network to run, including:
--     - Feed forward
--     - Back propagation
--     - Calculate Mean Square Error
]]

function linearFeedForward(weight, bias, input)
    local output = 0

    -- Weighted sum
    for i = 1, #weight do
        output = output + weight[i]*input[i]
    end

    -- Bias neuron
    output = output + bias

    return output
end


function activationFunction(x)
    return math.tanh(x)
end


function backPropagation()

end


function MSE()

end


-- We define this function in case we use it as 
-- our activation function
function sigmoid(x)
    return (1 / (1 + math.exp(-x)))
end

print(activationFunction(15))