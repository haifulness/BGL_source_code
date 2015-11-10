--[[
-- Author: Hai Tran
-- Date: Nov 09, 2015
-- Filename: bgl_dataLoading.lua
-- Descriptiion: This file tests the ability to call a function from another file in Lua.
--]]

require("../bgl_dataLoading.lua")
local path = "../../Datasets/Peter Kok - Real data for predicting blood glucose levels of diabetics/data.txt"
local date, time, glucose, shortActingInsulin, longActingInsulin, food, exercise, stress = loadFile(path)
print(date[1])