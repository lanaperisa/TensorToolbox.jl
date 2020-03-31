module SparseTensors

# Inspired by:
#
# BW Bader and TG Kolda. Efficient MATLAB Computations with Sparse
# and Factored Tensors, SIAM J Scientific Computing 30:205-231, 2007.
# <a href="http:dx.doi.org/10.1137/060676489"
# >DOI: 10.1137/060676489</a>. <a href="matlab:web(strcat('file://',...
# fullfile(getfield(what('tensor_toolbox'),'path'),'doc','html',...
# 'bibtex.html#TTB_Sparse')))">[BibTeX]</a>

mutable struct SparseTensor
    # tensor_toolbox style implementation
    # vals::Array{T,1}
    # subs::Array{Tuple{Int,Vararg{Int},1}}
    # dict::Dict{Tuple{Int,Vararg{Int}},T} # Julia doesn't like this
    # dict::Dict{NTuple{3,Int},T}
    vals::Array{T,1} where T
    subs::Array{UInt,2} # each row is the index of an entry
    dims::CartesianIndex
    function SparseTensor(v,subs)
        newsubs = unique(subs,dims=1)

        # This feels like it should be slow
        # c.f. dictionary implementation?
        newinds = [findfirst(r -> r==myr, eachrow(newsubs)|>collect) for myr in eachrow(subs)]
        vals = zeros(eltype(v),size(newsubs)[1])
        for i in 1:length(v)
            vals[newinds[i]] += v[i]
        end

        dims = CartesianIndex(maximum(newsubs, dims=1)...)
        new(vals,newsubs,dims)
    end
end

# Dict-based implementation is much faster + memory efficient
# TODO: work out how to mark vararg work
mutable struct SparseTensorD
    # tensor_toolbox style implementation
    # vals::Array{T,1}
    # subs::Array{Tuple{Int,Vararg{Int},1}}
    # dict::Dict{Tuple{Int,Vararg{Int}},T} # Julia doesn't like this
    dict::Dict{NTuple{3,UInt},T} where T
    #dims::CartesianIndex
    function SparseTensorD(v::Array{N,1},subs) where N <: Number
        newsubs = unique(subs,dims=1)

        # This feels like it should be slow
        dict = Dict(v => 0.0 for v in Tuple.(eachrow(newsubs)))
        for i in 1:length(v)
            dict[Tuple(subs[i,:])] += v[i]
        end

        #dims = CartesianIndex(maximum(newsubs, dims=1)...)
        new(dict)#,dims)
    end
    SparseTensorD(d) = new(d)
end


function Base.:+(l::SparseTensor, r::SparseTensor)
    SparseTensor([l.vals; r.vals], [l.subs; r.subs])
end

# TODO: define broadcasting
# See
# https://docs.julialang.org/en/v1/manual/interfaces/index.html
function Base.:+(l::SparseTensorD, r::SparseTensorD)
    SparseTensorD(merge(+,l.dict,r.dict))
end

function Base.getindex(A::SparseTensor, inds...)
    all(Tuple(A.dims) .>= inds) # TODO: or throw index error
    something(findfirst(r -> Tuple(r) == inds, eachrow(A.subs) |> collect), 0)
end

function Base.getindex(A::SparseTensorD, inds...)
    get(A.dict,inds,0)
end

function randtensor(n,d,dims=(256,256,256))
    subs = round.((rand(Float64,n,d)) .* ([d for d in dims]' .-1)) .+ 1 .|> UInt
    vals = rand(n)
    SparseTensor(vals,subs)
end

function randtensorD(n,d,dims=(256,256,256))
    subs = round.((rand(Float64,n,d)) .* ([d for d in dims]' .-1)) .+ 1 .|> UInt
    vals = rand(n)
    SparseTensorD(vals,subs)
end

# Base.:* - see ttt.m
end
