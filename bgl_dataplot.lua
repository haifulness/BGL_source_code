--[[
-- Author: Hai Tran
-- Date: Nov 24, 2015
-- Filename: bgl_dataplot.lua
-- Descriptiion: Use gnuplot to graph the status of the given data.
]]

require 'gnuplot'
require("bgl_dataLoading2.lua")

gnuplot.setterm('x11')

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

-- Convert tables into tensors
local data_storage, data

--
-- Morning alone
--
data_storage = torch.Storage(#morning_date)
for i = 1, #morning_date do
	data_storage[i] =   morning_glucose[i]
end
data = torch.Tensor(data_storage)

gnuplot.pngfigure('morning_glucose.png')
gnuplot.title('Morning Glucose')
gnuplot.xlabel('Day')
gnuplot.ylabel('Glucose Level')
gnuplot.plot(data)
gnuplot.plotflush()

--
-- Afternoon alone
--
data_storage = torch.Storage(#afternoon_date)
for i = 1, #afternoon_date do
	data_storage[i] = afternoon_glucose[i]
end
data = torch.Tensor(data_storage)

gnuplot.pngfigure('afternoon_glucose.png')
gnuplot.title('Afternoon Glucose')
gnuplot.xlabel('Day')
gnuplot.ylabel('Glucose Level')
gnuplot.plot(data)
gnuplot.plotflush()

--
-- Evening alone
--
data_storage = torch.Storage(#evening_date)
for i = 1, #evening_date do
	data_storage[i] = evening_glucose[i]
end
data = torch.Tensor(data_storage)

gnuplot.pngfigure('evening_glucose.png')
gnuplot.title('Evening Glucose')
gnuplot.xlabel('Day')
gnuplot.ylabel('Glucose Level')
gnuplot.plot(data)
gnuplot.plotflush()

--
-- Night alone
--
data_storage = torch.Storage(#night_date)
for i = 1, #night_date do
	data_storage[i] = night_glucose[i]
end
data = torch.Tensor(data_storage)

gnuplot.pngfigure('night_glucose.png')
gnuplot.title('Night Glucose')
gnuplot.xlabel('Day')
gnuplot.ylabel('Glucose Level')
gnuplot.plot(data)
gnuplot.plotflush()

--
-- Altogether
--
local morning_storage, morning, afternoon_storage, afternoon, 
      evening_storage, evening, night_storage, night

morning_storage   = torch.Storage(#morning_date)
afternoon_storage = torch.Storage(#afternoon_date)
evening_storage   = torch.Storage(#evening_date)
night_storage     = torch.Storage(#night_date)
for i = 1, #morning_date do
	morning_storage[i]   = morning_glucose[i]
	afternoon_storage[i] = afternoon_glucose[i]
	evening_storage[i]   = evening_glucose[i]
	night_storage[i]     = night_glucose[i]
end
morning   = torch.Tensor(morning_storage)
afternoon = torch.Tensor(afternoon_storage)
evening   = torch.Tensor(evening_storage)
night     = torch.Tensor(night_storage)

gnuplot.pngfigure('glucose.png')
gnuplot.title('Glucose Data')
gnuplot.xlabel('Day')
gnuplot.ylabel('Glucose Level')
gnuplot.plot(
	{'Morning', morning}, 
	{'Afternoon', afternoon}, 
	{'Evening', evening}, 
	{'Night', night})
gnuplot.plotflush()
