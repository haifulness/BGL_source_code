require 'torch'
require 'nn'
require 'parallel'

parallel.nfork(4)

-- define process' code:
code = function()
   -- arbitrary code contained here
   t = torch.Tensor(10)
   print(t)

   -- any process can access its id, its parent's id [and children's id]
   print(parallel.id)
   --print(parallel.parent.id)
   if parallel.children[1] then print(parallel.children[1].id) end
end

-- execute code in given process(es), with optional arguments:
parallel.children:exec(code)

-- this is equivalent to:
--for _,child in ipairs(parallel.child) do
--    child:exec(code)
--end