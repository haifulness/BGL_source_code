--[[
-- Author: Hai Tran
-- Date: Nov 16, 2015
-- Filename: fileLoadingTest3.lua
-- Descriptiion: This file tests bgl_dataLoading3.lua
--]]

require 'torch'
require 'nn'
require 'gnuplot'
require("bgl_dataLoading.lua")
require("bgl_generateSets.lua")

-- Neural net model:
--   + Linear
--   + 4 inputs
--   + 1 hidden layer with 10 nodes
--   + 1 output
--
local SIZE_INPUT = 4
local SIZE_HIDDEN_LAYER = 200
local SIZE_OUTPUT = 1

-- Load data
local path = "../Datasets/Peter Kok - Real data for predicting blood glucose levels of diabetics/data.txt"
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

print(morning_time)