--[[
-- Author: Hai Tran
-- Date: Nov 10, 2015
-- Filename: fileLoadingTest2.lua
-- Descriptiion: This file tests bgl_dataLoading2.lua
--]]

require("../bgl_dataLoading2.lua")
local path = "../../Datasets/Peter Kok - Real data for predicting blood glucose levels of diabetics/data.txt"
local 
	morning_date, 
    morning_time, 
    morning_glucose, 
    morning_SAI,  -- short acting insulin 
    morning_LAI,  -- long acting insulin
    morning_food, 
    morning_exercise, 
    morning_stress,
    
    afternoon_date, 
    afternoon_time, 
    afternoon_glucose, 
    afternoon_SAI,  -- short acting insulin 
    afternoon_LAI,  -- long acting insulin 
    afternoon_food, 
    afternoon_exercise, 
    afternoon_stress,

    evening_date, 
    evening_time, 
    evening_glucose, 
    evening_SAI,  -- short acting insulin  
    evening_LAI,  -- long acting insulin 
    evening_food, 
    evening_exercise, 
    evening_stress,

    night_date, 
    night_time, 
    night_glucose, 
    night_SAI,  -- short acting insulin  
    night_LAI,  -- long acting insulin 
    night_food, 
    night_exercise, 
    night_stress  

	= loadFile(path)

print(morning_time[0])