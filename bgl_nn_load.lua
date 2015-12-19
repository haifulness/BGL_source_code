--[[
-- Author: Hai Tran
-- Date: Dec 18, 2015
-- Filename: bgl_nn_load.lua
-- Description: Load the weights of a neural network from a txt file.
-- 
-- ]]

-- ====================
--
-- Load data from file
--
-- ====================

function loadNet(fileLocation)
    local file = io.open(fileLocation)

    local morning_date = { }

    if file then
    	local lineCounter = 1

    	-- Iterate through every line of the data file and insert values into the arrays
    	for line in file:lines() do

            -- It's strange here. I always received an integer when I divide lineCounter by 2,
            -- but in this case, I have to call math.ceil to round the division.
            local i = math.ceil(lineCounter / 8)

            if lineCounter % 8 == 2 then
        		morning_date[i], 
                morning_time[i], 
                morning_glucose[i], 
                morning_SAI[i], 
                morning_LAI[i], 
                morning_food[i], 
                morning_exercise[i], 
                morning_stress[i] 
                = unpack(line:split(";"))
    		end

            if lineCounter % 8 == 4 then
                afternoon_date[i], 
                afternoon_time[i], 
                afternoon_glucose[i], 
                afternoon_SAI[i], 
                afternoon_LAI[i], 
                afternoon_food[i], 
                afternoon_exercise[i], 
                afternoon_stress[i] 
                = unpack(line:split(";"))
            end

            if lineCounter % 8 == 6 then
                evening_date[i], 
                evening_time[i], 
                evening_glucose[i], 
                evening_SAI[i], 
                evening_LAI[i], 
                evening_food[i], 
                evening_exercise[i], 
                evening_stress[i] 
                = unpack(line:split(";"))
            end

            if lineCounter % 8 == 0 then
                night_date[i], 
                night_time[i], 
                night_glucose[i], 
                night_SAI[i], 
                night_LAI[i], 
                night_food[i], 
                night_exercise[i], 
                night_stress[i] 
                = unpack(line:split(";"))
            end

            -- Go to the next line
            lineCounter = lineCounter + 1
    	end
        
    else
    	print("Cannot open file") 
    end

    -- It's interesting that Lua can return multiple values like this. Very convenient.
    return 
        morning_date, 
        morning_time, 
        morning_glucose, 
        morning_SAI,  -- short acting insulin 
        morning_LAI,  -- long acting insulin
        morning_food, 
        morning_exercise, 
        morning_stress,
        ----
        afternoon_date, 
        afternoon_time, 
        afternoon_glucose, 
        afternoon_SAI,  -- short acting insulin 
        afternoon_LAI,  -- long acting insulin 
        afternoon_food, 
        afternoon_exercise, 
        afternoon_stress,
        ----
        evening_date, 
        evening_time, 
        evening_glucose, 
        evening_SAI,  -- short acting insulin  
        evening_LAI,  -- long acting insulin 
        evening_food, 
        evening_exercise, 
        evening_stress,
        ----
        night_date, 
        night_time, 
        night_glucose, 
        night_SAI,  -- short acting insulin  
        night_LAI,  -- long acting insulin 
        night_food, 
        night_exercise, 
        night_stress 
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
