# Inspired by:
#
# BW Bader and TG Kolda. Efficient MATLAB Computations with Sparse
# and Factored Tensors, SIAM J Scientific Computing 30:205-231, 2007.
# <a href="http:dx.doi.org/10.1137/060676489"
# >DOI: 10.1137/060676489</a>. <a href="matlab:web(strcat('file://',...
# fullfile(getfield(what('tensor_toolbox'),'path'),'doc','html',...
# 'bibtex.html#TTB_Sparse')))">[BibTeX]</a>

export SparseTensor, ttt, sptenrand

mutable struct SparseTensor{T,N} <: AbstractArray{T,N}
    dict::Dict{NTuple{N,Int},T}
    # dict::Dict{Tuple{Int,Vararg{Int}},T} # Julia doesn't like this
    dims::NTuple{N,Int}
    function SparseTensor(v::Array{E,1},subs::Array{Int,2},dims="auto") where E <: Number
        newsubs = unique(subs,dims=1)

        # This feels like it should be slow
        dict = Dict(v => 0.0 for v in Tuple.(eachrow(newsubs)))
        for i in 1:length(v)
            dict[Tuple(subs[i,:])] += v[i]
        end

        #dims = CartesianIndex(maximum(newsubs, dims=1)...)
        SparseTensor(dict,dims == "auto" ? maximum(subs,dims=1) |> Tuple : dims)
    end
    SparseTensor(d::Dict,dims) = new{d |> values |> eltype, length(dims)}(d,dims)
    SparseTensor(d::Dict) = SparseTensor(d,maximum(_indarray(d),dims=1) |> Tuple)
    SparseTensor(a::AbstractArray) = SparseTensor(Dict(Tuple(ind) => a[ind] for ind in findall(x->x!==zero(x),a)),size(a))
end

function _indarray(d::Dict)
    reduce(hcat,d |> keys .|> collect) |> permutedims
end
_indarray(t::SparseTensor) = _indarray(t.dict)

Base.size(t::SparseTensor) = t.dims

# TODO: add error checking - out of bounds etc.
function Base.getindex(A::SparseTensor{T,N}, inds::Vararg{Int,N}) where {N,T}
    get(A.dict,inds,zero(eltype(A)))
end
# Bug: Julia asks for (1,1) indices for rank-1 tensors - this is a workaround but not a pretty one
# and this doesn't seem to do anything
# Base.IndexStyle(::SparseTensor{T,1}) where T = IndexLinear()
function Base.getindex(A::SparseTensor{T,1}, inds::Vararg{Int,N}) where {N,T}
    get(A.dict,(inds[1],),zero(eltype(A)))
end

function Base.setindex!(A::SparseTensor, value, inds::Vararg{Int,N}) where N
    A.dict[inds] = value
end

function sptenrand(dims::Array{Int,1},n::Int)
    subs = round.((rand(Float64,n,length(dims))) .* (dims' .-1)) .+ 1 .|> Int
    vals = rand(n)
    SparseTensor(vals,subs,dims|>Tuple)
end

sptenrand(dims,d::AbstractFloat) = sptenrand(dims,d*prod(dims)|>round|>Int)

# Broadcasting - inspired by https://github.com/andyferris/Dictionaries.jl/blob/9b22a254683260354fda8e32291727dfad040106/src/broadcast.jl#L94
# See also: https://docs.julialang.org/en/v1/manual/interfaces/index.html
Base.Broadcast.broadcastable(t::SparseTensor) = t

struct SparseTensorBroadcastStyle{N} <: Base.Broadcast.AbstractArrayStyle{N}
end

Base.Broadcast.BroadcastStyle(::Type{<:SparseTensor{T,N}}) where {T,N} = SparseTensorBroadcastStyle{N}()

# Fall back to dense array if one of the broadcast arguments is a dense array
Base.Broadcast.BroadcastStyle(::SparseTensorBroadcastStyle{D}, a::Base.Broadcast.DefaultArrayStyle{N}) where {D,N} = Base.Broadcast.DefaultArrayStyle{max(N,D)}()

# TODO: 
# - sparse .* dense = sparse 
# - sparse .+ dense = dense, but it can be sped up by being s .+ d = begin c = copy(d); for (k,v) in s.dict; c[k...] .+ v; end
# (need to make own SparseDense broadcast style to replace the DefaultArrayStyle above and then make the function below specialise on f={Base.:*,Base.:+}

function Base.Broadcast.broadcasted(::SparseTensorBroadcastStyle, f, args...)
    merge(f,_getdict.(args)...) |> SparseTensor
end

@inline _getdict(t::SparseTensor) = t.dict

# I'm not sure what these do but they're not good
# Base.Broadcast.materialize(t::SparseTensor) = copy(t)
# Base.Broadcast.materialize!(out::SparseTensor, t::SparseTensor) = copyto!(out, t)


# Base.:* - see ttt.m
# Outer product: (l⊗r)_{i_1,...,i_n,j_1,...,j_n} = l_{i_1,...,i_n} r_{j_1,...,j_n} (i.e. not the Kronecker product)
function ttt(l::SparseTensor, r::SparseTensor)
    SparseTensor(Dict((a..., b...) => l[a...] * r[b...] for (a, b) in Iterators.product(l.dict |> keys,r.dict |> keys)),(l.dims...,r.dims...))
end

function Base.Array(t::SparseTensor)
    a = zeros(size(t))
    for (k,v) in t.dict
        a[k...] = v
    end
    a
end

function Base.:*(t::SparseTensor{T,2},v::AbstractArray{T2,1}) where {T,T2}
    density = (t.dict |> length) / (t.dims |> prod)

    # Heuristic: back off if density means this is slower than base multiply
    (density > 2e-4) && return Array(t)*v
    inds = _indarray(t)
    SparseTensor(Dict((k,) => sum(t[r...]*v[r[2]] for r in eachrow(inds[inds[:,1].==k,:]))
                      for k in inds[:,1] |> unique),size(v))
end

# TODO: Support slices / Colon()

# TODO: stop implementing these myself. I should just write a TensorToolbox.jl compatible type and then optimise when I can be bothered. _headdesk_
# (bonus: it would let me check my maths)
