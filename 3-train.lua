--[[
-- Author: Hai Tran
-- Date: Jul 15, 2016
-- File: 3-train.lua
--]]


--------------------------------------------------------------------------------
-- Train a model
--
function train(model, train_input, train_output)
	model: training()
	local w, grads = model:getParameters()
	local total_err = 0

	----------------------------------------------------------------------------
	-- Define eval closure
	--
	local feval = function(w_new)
		-- Reset data
		if w ~= w_new then w:copy(w_new) end
		-- Reset gradients
		grads:zero()

		idx = (idx or 0) + 1
		if idx > train_input:size(1) then idx = 1 end

		local input = train_input[idx]
		local output = torch.Tensor(1)
		output[1] = train_output[idx]

		local prediction = model:forward(input)
		local err = criterion:forward(prediction, output)
		local df_dw = criterion:backward(prediction, output)
		model:backward(input, df_dw)

		total_err = total_err + err

		-- Return 
		return  err, grads
	end

	----------------------------------------------------------------------------
	-- Train
	--

	for j = 1, train_input:size(1) do
		w, fs = optim.sgd(feval, w, optimState)
	end

	-- Report average error on epoch
	total_err = total_err / train_input:size(1)

	return total_err
end


--------------------------------------------------------------------------------
-- Test a model
--
function test(model, test_input, test_output)
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

	return total_err/test_input:size(1)
end
