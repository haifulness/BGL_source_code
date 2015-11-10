--[[
-- Author: Hai Tran
-- Date: Nov 09, 2015
-- Filename: bgl_dataLoading.lua
-- Description: Load data from the given text file and store it in different arrays. This file has 2 functions:
--   1. loadFile(path): Load data from a file given its location. Each line of the loaded file will then stored
--      in different arrays representing 8 aspects of each interval of a day:
--        + Date of measurement
--        + Time of measurement
--        + Current glucose level
--        + Amount of short acting insulin
--        + Amount of long acting insulin
--        + Amount of food intake
--        + Amount of exercise (from 1 to 5)
--        + Level of stress
--
--   2. split(delimiter)
--
-- ]]

-- ====================
--
-- Load data from file
--
-- ====================

function loadFile(fileLocation)

    local file = io.open(fileLocation)

    -- 8 aspects to extract from each line of the file
    local date, time, glucose, shortActingInsulin, longActingInsulin, food, exercise, stress 
        = { }, { }, { }, { }, { }, { }, { }, { }

    if file then
    	local lineCounter = 1

    	-- Iterate through every line of the data file and insert values into the arrays
    	for line in file:lines() do
            -- Only load even lines, so we don't have any empty value
            if lineCounter % 2 == 0 then
                local i = lineCounter / 2
        		date[i], time[i], glucose[i], shortActingInsulin[i], longActingInsulin[i], food[i], exercise[i], stress[i] = unpack(line:split(";"))
    		end
            lineCounter = lineCounter + 1
    	end
        
    else
    	print("Cannot open file") 
    end

    -- It's interesting that Lua can return multiple values like this. Very convenient.
    return date, time, glucose, shortActingInsulin, longActingInsulin, food, exercise, stress
end



-- ====================
--
-- Split a string 
-- The following code is derived from
--   http://stackoverflow.com/questions/19262761/lua-need-to-split-at-comma
--
-- ====================

function string:split(delimiter, returnResult)

    -- In Lua, a function can be called without being given enough parameters.
    -- In the case of this function, we can call stringA:split(';') -- it's
    -- totally fine. However, that function call will not result in anything
    -- if we don't include the below if statement. The reason is returnResult
    -- is not initiated; the function has no address to return its values to.
    if not returnResult then
        returnResult = { }
    end

    -- A pointer that iterates along the given string
    local currPtr = 1

    -- Indicates the first and the last letters of the substring
    local subStringBegin, subStringEnd = string.find(self, delimiter, currPtr)
    
    -- Keep iterating while the string.find() function still returns a non-nil value
    while subStringBegin do
        
        -- Cut the substring from the current position and insert it into returnResult.
        -- If a substring is empty, the value we put into returnResult will be 0.
        local sub = string.sub(self, currPtr, subStringBegin-1)
        if sub == nil or sub == '' then
           sub = 0
        end
        table.insert(returnResult, sub)

        -- Move everything forward
        currPtr = subStringEnd + 1
        subStringBegin, subStringEnd = string.find(self, delimiter, currPtr)
    end

    -- Insert the last piece of the string into returnResult
    table.insert(returnResult, string.sub(self, currPtr))

    return returnResult
end
