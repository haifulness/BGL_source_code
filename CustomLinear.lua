--[[
-- Original: https://groups.google.com/forum/#!topic/torch7/kTIs66zvFhs
--]]

require 'nn'

do
    local Linear, parent = torch.class('nn.CustomLinear', 'nn.Linear')
    
    -- override the constructor to have the additional range of initialization
    function Linear:__init(inputSize, outputSize, mean, std)
        parent.__init(self,inputSize,outputSize)
                
        self:reset(mean,std)
    end
    
    -- override the :reset method to use custom weight initialization.        
    function Linear:reset(mean,stdv)
        
        if mean and stdv then
            self.weight:normal(mean,stdv)
            self.bias:normal(mean,stdv)
        else
            self.weight:normal(0,1)
            self.bias:normal(0,1)
        end
    end

end