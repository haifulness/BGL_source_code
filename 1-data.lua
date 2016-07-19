--[[
-- Author: Hai Tran
-- Date: Jul 15, 2016
-- File: 1-data.lua
--]]


--------------------------------------------------------------------------------
-- Global variables
-- 
raw_data = {}

-------------------------------------------------------------------------------
-- Constants.
--
INPUT_SIZE = 23
DATASET_SIZE = 305


--------------------------------------------------------------------------------
-- Load data
-- 
function load_data(path)
	local file = io.open(path, "r")

	if file ~= nil then
		local counter = 0

		while true do
			local line = file:read()
			line = file:read()
			if line == nil then break end

			counter = counter + 1
			raw_data[counter] = {}

			raw_data[counter][1],  -- date
			raw_data[counter][2],  -- time
			raw_data[counter][3],  -- BGL
			raw_data[counter][4],  -- SAI
			raw_data[counter][5],  -- LAI
			raw_data[counter][6],  -- meal
			raw_data[counter][7],  -- exercise
			raw_data[counter][8]   -- stress
				= line:match("([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);")

			--[[
			-- Check for blank information
			if raw_data[counter][1] == "" then print(counter, 1) end
			if raw_data[counter][2] == "" then print(counter, 2) end
			if raw_data[counter][3] == "" then print(counter, 3) end
			if raw_data[counter][4] == "" then print(counter, 4) end
			if raw_data[counter][5] == "" then print(counter, 5) end
			if raw_data[counter][6] == "" then print(counter, 6) end
			if raw_data[counter][7] == "" then print(counter, 7) end
			if raw_data[counter][8] == "" then print(counter, 8) end
			-- Results: only missing [4] & [5] (insulin shots) in some entries.
			-- By default, we'll consider these missing values as 0.
			--]]

			--
			-- Time is required
			if raw_data[counter][2] == nil then 
				counter = counter - 1
			else
				-- Extract time (string) into hour and minute
				local hour, min = string.match(raw_data[counter][2], "(%d+):(%d+)")

				-- Convert everything from string to number
				raw_data[counter][2] = tonumber(hour) + tonumber(min)/60
				raw_data[counter][3] = tonumber(raw_data[counter][3])
				raw_data[counter][4] = tonumber(raw_data[counter][4])
				raw_data[counter][5] = tonumber(raw_data[counter][5])
				raw_data[counter][6] = tonumber(raw_data[counter][6])
				raw_data[counter][7] = tonumber(raw_data[counter][7])
				raw_data[counter][8] = tonumber(raw_data[counter][8])

				-- Set missing values as 0
				if raw_data[counter][4] == nil then raw_data[counter][4] = 0 end
				if raw_data[counter][5] == nil then raw_data[counter][5] = 0 end
			end
		end

		io.close(file)
		io.flush()
	end
end


--------------------------------------------------------------------------------
-- Generate input & target output
-- 
function gen_data()
	local data = torch.Tensor(DATASET_SIZE, INPUT_SIZE)
	local target = torch.Tensor(DATASET_SIZE)

	for index = 4, DATASET_SIZE + 3 do
		-- Target output
		target[index-3] = raw_data[index][3]

		-- Input
		--
		-- Current interval: SAI
		data[index-3][1] = raw_data[index][4]
		-- Current interval: LAI
		data[index-3][2] = raw_data[index][5]
		-- Current interval: meal
		data[index-3][3] = raw_data[index][6]
		-- Current interval: exercise
		data[index-3][4] = raw_data[index][7]
		-- Current interval: stress
		data[index-3][5] = raw_data[index][8]

		-- 1st prior interval: BGL
		data[index-3][6] = raw_data[index-1][3]
		-- 1st prior interval: SAI
		data[index-3][7] = raw_data[index-1][4]
		-- 1st prior interval: LAI
		data[index-3][8] = raw_data[index-1][5]
		-- 1st prior interval: meal
		data[index-3][9] = raw_data[index-1][6]
		-- 1st prior interval: exercise
		data[index-3][10] = raw_data[index-1][7]
		-- 1st prior interval: stress
		data[index-3][11] = raw_data[index-1][8]

		-- 2nd prior interval: BGL
		data[index-3][12] = raw_data[index-2][3]
		-- 2nd prior interval: SAI
		data[index-3][13] = raw_data[index-2][4]
		-- 2nd prior interval: LAI
		data[index-3][14] = raw_data[index-2][5]
		-- 2nd prior interval: meal
		data[index-3][15] = raw_data[index-2][6]
		-- 2nd prior interval: exercise
		data[index-3][16] = raw_data[index-2][7]
		-- 2nd prior interval: stress
		data[index-3][17] = raw_data[index-2][8]

		-- 3rd prior interval: BGL
		data[index-3][18] = raw_data[index-3][3]
		-- 3rd prior interval: SAI
		data[index-3][19] = raw_data[index-3][4]
		-- 3rd prior interval: LAI
		data[index-3][20] = raw_data[index-3][5]
		-- 3rd prior interval: meal
		data[index-3][21] = raw_data[index-3][6]
		-- 3rd prior interval: exercise
		data[index-3][22] = raw_data[index-3][7]
		-- 3rd prior interval: stress
		data[index-3][23] = raw_data[index-3][8]
	end

	return data, target
end

--------------------------------------------------------------------------------
-- Test
--[[
load_data("data/data.txt")
data, target = gen_data()

for i = 1, DATASET_SIZE do
	for j = 1, INPUT_SIZE do
		if type(data[i][j]) ~= "number" then print(i, j) end
	end
end
--]]
