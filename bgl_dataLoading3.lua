--[[
-- Author: Hai Tran
-- Date: Nov 16, 2015
-- Filename: bgl_dataLoading3.lua
-- Description: Load data from the given text file and store it in different arrays. This file has 2 functions:
--   1. loadFile(path): 
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
    local loaded = torch.load(fileLocation, 'ascii')
    print(loaded.data)
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
