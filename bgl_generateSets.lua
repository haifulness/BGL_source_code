--[[
-- Author: Hai Tran
-- Date: Nov 23, 2015
-- Filename: bgl_generateSets.lua
-- Description: From a given dataset, generate three subsets: Train, Test, 
--   and Validate based on the given percentages (how many elements should
--   each set contains).
-- Solution: Instead of touching the dataset, we will only shuffle the list
--   of indices then divide the list into three sublists.
]]

-- Seed the random function
math.randomseed(os.time())

--[[
-- Shuffle a list
]]
function shuffle(list)
	for i = 1, #list do
		local j, k = math.random(#list), math.random(#list)
		list[j], list[k] = list[k], list[j]
	end
	return list
end

--[[
-- Get a subrange of a list
]]
function subrange(list, beginIndex, endIndex)
	local sub = {}
	for i = beginIndex, endIndex do
		sub[#sub + 1] = list[i]
	end
	return sub
end

--[[
-- Return indices of each set
]]
function generateSets(size, trainPercentage, testPercentage, validatePercentage)
	local train, test, validate = {},{},{}

	-- Size of each subset. They should add up to size.
	local trainSize    = math.floor(size * trainPercentage / 100)
	local testSize     = math.floor(size * testPercentage / 100) 
	local validateSize = math.floor(size * validatePercentage / 100)

	-- Generate the list of indices
	local indices = {}
	for i = 1, size do
		indices[i] = i
	end

	-- Shuffle the list
	indices = shuffle(indices)

	-- Assign subsets
	train    = subrange(indices, 1, trainSize)
	test     = subrange(indices, trainSize + 1, trainSize + testSize)
	validate = subrange(indices, trainSize + testSize + 1, size)

	return train, test, validate
end
