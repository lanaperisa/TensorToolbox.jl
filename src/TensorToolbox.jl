#v0.7
module TensorToolbox

#Tensors in Tucker format + functions

import LinearAlgebra: norm, rank
import Base: +, -, *, .*, ==, cat, display, full, isequal, kron, ndims, parent, permutedims, show, size, squeeze

include("helper.jl")
include("tensor.jl")
include("ttensor.jl")
include("ktensor.jl")
include("dimtree.jl")
include("htensor.jl")

end #module
