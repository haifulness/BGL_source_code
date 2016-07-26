--[[
-- Author: Hai Tran
-- Date: Jul 15, 2016
-- File: 3-train.lua
--]]


--------------------------------------------------------------------------------
-- Train
--
function train(train_input, train_output)
	model:training()
	local w, grads = model:getParameters()
	local trainErr = 0

	for idx = 1, train_input:size(1) do

		-- Define eval closure
		local feval = function(w_new)
			-- Reset data
			if w ~= w_new then w:copy(w_new) end
			-- Reset gradients
			grads:zero()

			local input = train_input[idx]
			local output = torch.Tensor(1)
			output[1] = train_output[idx]

			local prediction = model:forward(input)
			local err = criterion:forward(prediction, output)
			local df_dw = criterion:backward(prediction, output)
			model:backward(input, df_dw)

			trainErr = trainErr + err

			return  err, grads
		end

		-- Train
		optim.sgd(feval, w, optimState)
	end

	return trainErr/train_input:size(1)
end


--------------------------------------------------------------------------------
-- Validate
--
function validate(validate_input, validate_output)
	model:training()
	local w, grads = model:getParameters()
	local validateErr = 0

	for idx = 1, validate_input:size(1) do

		-- Define eval closure
		local feval = function(w_new)
			-- Reset data
			if w ~= w_new then w:copy(w_new) end
			-- Reset gradients
			grads:zero()

			local input = validate_input[idx]
			local output = torch.Tensor(1)
			output[1] = validate_output[idx]

			local prediction = model:forward(input)
			local err = criterion:forward(prediction, output)
			local df_dw = criterion:backward(prediction, output)
			model:backward(input, df_dw)

			validateErr = validateErr + err

			return  err, grads
		end

		-- Train
		optim.sgd(feval, w, optimState)
	end

	return validateErr/validate_input:size(1)
end


--------------------------------------------------------------------------------
-- Test
--
function test(test_input, test_output)
	model:evaluate()
	local total_err = 0

	for idx = 1, test_input:size(1) do
		local input = test_input[idx]
		local output = torch.Tensor(1)
		output[1] = test_output[idx]

		local prediction = model:forward(input)
		local err = criterion:forward(prediction, output)
		total_err = total_err + err
	end

	model:training()

	return total_err/test_input:size(1)
end
