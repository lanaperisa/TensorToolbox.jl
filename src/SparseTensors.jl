module SparseTensors

# Inspired by:
#
# BW Bader and TG Kolda. Efficient MATLAB Computations with Sparse
# and Factored Tensors, SIAM J Scientific Computing 30:205-231, 2007.
# <a href="http:dx.doi.org/10.1137/060676489"
# >DOI: 10.1137/060676489</a>. <a href="matlab:web(strcat('file://',...
# fullfile(getfield(what('tensor_toolbox'),'path'),'doc','html',...
# 'bibtex.html#TTB_Sparse')))">[BibTeX]</a>

mutable struct SparseTensor{T,N} <: AbstractArray{T,N}
    dict::Dict{NTuple{N,Int},T}
    # dict::Dict{Tuple{Int,Vararg{Int}},T} # Julia doesn't like this
    #dims::CartesianIndex
    function SparseTensor(v::Array{E,1},subs) where E <: Number
        newsubs = unique(subs,dims=1)

        # This feels like it should be slow
        dict = Dict(v => 0.0 for v in Tuple.(eachrow(newsubs)))
        for i in 1:length(v)
            dict[Tuple(subs[i,:])] += v[i]
        end

        #dims = CartesianIndex(maximum(newsubs, dims=1)...)
        SparseTensor(dict)
    end
    SparseTensor(d) = new{d |> values |> eltype, d |> keys |> first |> length}(d)
end

function Base.size(t::SparseTensor)
    # Perhaps we should just store this at creation time - cheaper to work it out on subs?
    maximum(hcat(([k...] for k in keys(t.dict))...),dims=2) |> Tuple
end


function Base.:+(l::SparseTensor, r::SparseTensor)
    SparseTensor([l.vals; r.vals], [l.subs; r.subs])
end

function Base.:+(l::SparseTensor, r::SparseTensor)
    SparseTensor(merge(+,l.dict,r.dict))
end

function Base.getindex(A::SparseTensor, inds::Vararg{Int,N}) where N # this should really be where N == ndims(A)
    get(A.dict,inds,0)
end

function Base.setindex!(A::SparseTensor, value, inds::Vararg{Int,N}) where N
    A.dict[inds] = value
end

function randtensor(n,dims=(256,256,256))
    subs = vcat(round.((rand(Float64,n,length(dims))) .* ([d for d in dims]' .-1)) .+ 1 .|> Int, [Int(d) for d in dims]')
    vals = [rand(n); 0.0]
    SparseTensor(vals,subs)
end

# Broadcasting - inspired by https://github.com/andyferris/Dictionaries.jl/blob/9b22a254683260354fda8e32291727dfad040106/src/broadcast.jl#L94
# See also: https://docs.julialang.org/en/v1/manual/interfaces/index.html
Base.Broadcast.broadcastable(t::SparseTensor) = t

struct SparseTensorBroadcastStyle <: Base.Broadcast.BroadcastStyle
end

Base.Broadcast.BroadcastStyle(::Type{<:SparseTensor}) = SparseTensorBroadcastStyle()

# TODO: fix SparseTensor(...) .+ {Numbers, AbstractArrays (where indices match)
# Base.Broadcast.BroadcastStyle(::SparseTensorBroadcastStyle, ::Base.Broadcast.AbstractArrayStyle{0}) = SparseTensorBroadcastStyle()

function Base.Broadcast.broadcasted(::SparseTensorBroadcastStyle, f, args...)
    merge(f,_getdict.(args)...) |> SparseTensor
end

@inline _getdict(t::SparseTensor) = t.dict

# I'm not sure what these do but they're not good
# Base.Broadcast.materialize(t::SparseTensor) = copy(t)
# Base.Broadcast.materialize!(out::SparseTensor, t::SparseTensor) = copyto!(out, t)


# Base.:* - see ttt.m
end
