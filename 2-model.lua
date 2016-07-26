--[[
-- Author: Hai Tran
-- Date: Jul 15, 2016
-- File: 2-model.lua
--]]


--------------------------------------------------------------------------------
-- Build a neural network using a function so we can generate it autiomatically
--
function buildModel(input_size, hidden_size, output_size, num_hidden_layers, activate_function)
	local net = nn.Sequential()

	-- Rescale the input tensor so the values all lie in the range (0, 1)
	-- and sum to 1.
	--net:add(nn.SoftMax())

	-- Input layer
	net:add(nn.Linear(input_size, hidden_size))

	if activate_function == "sigmoid" then
		net:add(nn.Sigmoid())
	else
		net:add(nn.Tanh())
	end

	-- Assume that all hidden layers have the same size
	for i = 1, num_hidden_layers - 1 do
		net:add(nn.Linear(hidden_size, hidden_size))

		if activate_function == "sigmoid" then
			net:add(nn.Sigmoid())
		else
			net:add(nn.Tanh())
		end
	end

	-- Re-adjust the output layer
	--[[
	if activate_function == "sigmoid" then
		net:add(nn.Sigmoid())
	else
		net:add(nn.Tanh())
	end
	--]]

	-- Output layer
	net:add(nn.Linear(hidden_size, output_size))	

	-- Criterion
	local crit = nn.MSECriterion()

	return net, crit
end


