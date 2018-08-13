#v0.6
module TensorToolbox

#Tensors in Tucker format + functions
using LinearAlgebra
import Base: +, -, *, .*, ==, cat, display, full, isequal, kron, ndims, parent, permutedims, show, size, squeeze

include("helper.jl")
include("tensor.jl")
include("ttensor.jl")
include("ktensor.jl")
include("dimtree.jl")
include("htensor.jl")

end #module
