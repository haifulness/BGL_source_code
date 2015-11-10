--[[
-- Author: Hai Tran
-- Date: Nov 08, 2015
-- Filename: bgl_dataLoading.lua
-- Description: This file tests the use of nngraph. 
-- Output: file "nngraph_sample.svg" in the same folder
--]]



-- ====================
--
-- Split a string 
-- The following code is obtained from
--   http://stackoverflow.com/questions/19262761/lua-need-to-split-at-comma
-- 
-- Modifications are stated in the code.
--
-- ====================
function string:split( inSplitPattern, outResults )
    if not outResults then
        outResults = { }
    end

    local theStart = 1
    local theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart )
    while theSplitStart do
    	
    	-- The below part was modified so if a substring is empty, the value we put
    	-- into outResults will be 0 instead of nil
    	local sub = string.sub( self, theStart, theSplitStart-1 )
    	if sub == nil or sub == '' then
    	   sub = 0
        end
        table.insert( outResults, sub )
        -- END MODIFICATION

        theStart = theSplitEnd + 1
        theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart )
    end

    table.insert( outResults, string.sub( self, theStart ) )

    return outResults
end



-- ====================
--
-- Load data from file
--
-- ====================
local path = "../../Datasets/Peter Kok - Real data for predicting blood glucose levels of diabetics/data.txt"
local file = io.open(path)
local date, time, glucose, shortActingInsulin, longActingInsulin, mealCarbohydrates, exercise, stress = { }, { }, { }, { }, { }, { }, { }, { }

if file then
	-- The index of the arrays, where we should put new data in
	local i = 0

	-- Iterate through every line of the data file and insert values into the arrays
	for line in file:lines() do
		date[i], time[i], glucose[i], shortActingInsulin[i], longActingInsulin[i], mealCarbohydrates[i], exercise[i], stress[i] = unpack(line:split(";"))
		i = i + 1
	end
else
	print("Cannot open file") 
end



-- ====================
--
-- Build a simple neural networks
--
-- ====================
require 'nn'
require 'nngraph'

-- Add layers
-- The first line creates the first module with #glucose inputs and 25 hidden nodes
-- The second line creates the output module. From 25 hidden nodes, the net generates 1 output.
-- The other lines add the modules to the net.
module1 = nn.Linear(#glucose, 25)()  -- the ending parantheses wrap the entire layer in a circle
module2 = nn.Linear(25, 1)(module1)  -- I don't yet understand why (module1) is needed here. It raises an error if I don't have it.
sampleNN = nn.gModule({module1}, {module2})

x = torch.Tensor(glucose)
dx = torch.rand(1)
sampleNN:updateOutput(x)
sampleNN:updateGradInput(x, dx)
sampleNN:accGradParameters(x, dx)


graph.dot(sampleNN.fg, 'nngraph_sample', 'nngraph_sample')

-- So, the weakness of nngraph:
--   + We don't really draw an existing net. Basically, we have to redo the steps of building a net using gModule.
--   + What we actually draw is just its wrapper. The image only displays how many nodes each layer
--     has. We cannot see the weights of each layer.
