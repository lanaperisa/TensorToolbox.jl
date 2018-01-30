#v0.6
module TensorToolbox

#Tensors in Tucker format + functions

import Base: +, -, *, .*, ==, display, full, isequal, ndims, normalize, normalize!, permutedims, show, size, vecnorm

include("helper.jl")
include("tensor.jl")
include("ttensor.jl")
include("ktensor.jl")
include("dimtree.jl")
include("htensor.jl")

end #module
