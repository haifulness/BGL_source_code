--
-- Hai Tran
-- Nov 02, 2015
--
-- This file is used for practicing using Torch7 and its supplemental 
-- components for Artificial Neural Networks such as nn, nngraph, etc.
--



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

-- As the first step, we do not care whether the below net has any meaning.
-- Write some working codes first!!!
--

-- Create a sequential net
simpleNN = nn.Sequential()

-- Add layers
-- The first line creates the first module with #glucose inputs and 25 hidden nodes
-- The second line creates the output module. From 25 hidden nodes, the net generates 1 output.
-- The other lines add the modules to the net.
module1 = nn.Linear(#glucose, 25)
module2 = nn.Linear(25, 1)
simpleNN:add(module1)
simpleNN:add(module2)

-- Print the net to see how it looks like
-- print("\nModule 1: ")
-- print(module1.weight)
-- print("\nModule 2: ")
-- print(module2.weight)
print(simpleNN)

-- Convert the data array to a tensor, put it into the net, 
-- apply feedforward on it, and print the output
print("\nOutput")
--print(simpleNN:forward(torch.Tensor(glucose)))



-- ====================
--
-- Use nngraph to draw the net
--
-- ====================
require 'nngraph'

-- Basically, we don't need the third parameter
-- But there is a problem between torch7 & nngraph.
-- Instead of display on the screen, we save it into a *.svg file

--[[
graph.dot(simpleNN.fg, 'SimpleNN', 'SimpleNN')
]]--

-- Oops, it's not this simple...
-- I'll read this paper to figure it out
--   https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/practicals/practical5.pdf



-- ====================
--
-- This part is to construct our own nn modules.
-- The official tutorial is here
--   http://torch.ch/docs/developer-docs.html
--
-- This tutorial looks easier to understand
--   https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/practicals/practical4.pdf
-- I'll look at them more closely
--
-- ====================
--[[
local NewClass, Parent = torch.class('nn.NewClass', 'nn.Module')

function NewClass:__init()
	Parent.__init(self)
end

function NewClass:updateOutput(input)
end

function NewClass:updateGradInput(input, gradOutput)
end

function NewClass:accGradParameters(input, gradOutput)
end

function NewClass:reset()
end
]]
