# TensorToolbox.jl

[![Build Status](https://travis-ci.org/lanaperisa/Tensors.jl.svg?branch=master)](https://travis-ci.org/lanaperisa/Tensors.jl)

Julia package for tensors. Includes functions for dense tensors and tensors in Tucker format. Follows the functionality of MATLAB [Tensor toolbox](http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html). 

## Basics

Start with
```julia
using TensorToolbox
```

Define tensor as multidimensional arrays and calculate its norm:
```julia
X=rand(4,3,2)
norm(X)
```
For two tensors of same size calculate their inner product:
```julia
X=rand(3,3,3,3);Y=rand(3,3,3,3);
innerprod(X,Y)
```
*n-mode matricization* of a tensor:
```julia
X=rand(4,3,2);n=1;
A=tenmat(X,n)
```
Fold matrix back to tensor:
```julia
X=matten(A,n,[4,3,2])
```
*n-mode product* of a tensor and a matrix or an array of matrices:
```julia
X=rand(5,4,3);
A=[rand(2,5),rand(2,4),rand(2,3)];
ttm(X,A[1],1)  #X times A[1] by mode 1
ttm(X,[A[1],A[2]],[1,2]) #X times A[1] by mode 1 and times A[2] by mode 2; same as ttm(X,A,-3)
ttm(X,A) #X times matrices from A by each mode
```
*n-mode (vector) product* of a tensor and a vector or an array of vectors:
```julia
X=rand(5,4,3);
V=[rand(5),rand(4),rand(3)];
ttv(X,V[1],1)  #X times V[1] by mode 1
ttv(X,[V[1],V[2]],[1,2]) #X times V[1] by mode 1 and times V[2] by mode 2; same as ttm(X,V,-3)
ttv(X,V) #X times vectors from V by each mode
```
*Outer product* of two tensors:
```julia
 X=rand(5,4,3,2);Y=rand(2,3,4);
 ttt(X,Y)
```
Kronecker product of two tensors (straightforward generalization of Kronecker product of matrices):
```julia
X=rand(5,4,3);Y=rand(2,2,2);
tkron(X,Y)
```
The *n-rank* and the *mutlilinear rank* of a tensor:
```julia
X=rand(5,4,3);
n=2;
nrank(X,n)
mrank(X)
```
The HOSVD:
```julia
X=rand(5,4,3);
hosvd(X) #same as T1=hosvd(X,eps_abs=1e-8)
hosvd(X,eps_abs=1e-6) #discard singular values lower than 1e-5
hosvd(X,eps_rel=1e-3) #discard singular values lower than 1e-3*sigma_{max}
hosvd(X,reqrank=[2,2,2])
```

## Tensors in Tucker format

Define tensor in Tucker format by its core tensor and factor matrices:
```julia
F=rand(5,4,3);
A=[rand(6,5),rand(6,4),rand(6,3)];
ttensor(F,A)
```
Get Tucker format of a tensor by using HOSVD:
```julia
X=rand(8,9,7);
hosvd(X) 
hosvd(X,reqrank=[3,3,3]) #HOSVD with predefined multilinear rank
```
Create random tensor in Tucker format of size 5x4x3 and mulilinear rank (2,2,2): 
```julia
X=randttensor([5,4,3],[2,2,2])
```
Basic functions:
```julia
size(X) 
coresize(X)
ndims(X)
norm(X)
full(X)  #Creates full tensor out of Tucker format
reorth(X) #Orthogonalize factor matrices
permutedims(X,[2,1,3]) 
```
*n-mode matricization* of a tensor in Tucker format:
```julia
n=1;
tenmat(X,n)
```
Basic operations:
```julia
X=randttensor([5,4,3],[2,2,2]);Y=randttensor([5,4,3],[3,2,1]);
innerprod(X,Y)
X+Y
X-Y
X==Y #same as isequal(X,Y)
3*X #same as mtimes(3,X)
```
*n-mode product* of a tensor in Tucker format and a matrix or an array of matrices:
```julia
X=randttensor([5,4,3],[2,2,2]);
A=[rand(2,5),rand(2,4),rand(2,3)];
ttm(X,A[1],1)  #X times A[1] by mode 1
ttm(X,[A[1],A[2]],[1,2]) #X times A[1] by mode 1 and times A[2] by mode 2; same as ttm(X,A,-3)
ttm(X,A) #X times matrices from A by each mode
```
*n-mode (vector) product* of a tensor in Tucker format and a vector or an array of vectors:
```julia
X=randttensor([5,4,3],[2,2,2]);
V=[rand(5),rand(4),rand(3)];
ttv(X,V[1],1)  #X times V[1] by mode 1
ttv(X,[V[1],V[2]],[1,2]) #X times V[1] by mode 1 and times V[2] by mode 2; same as ttm(X,V,-3)
ttv(X,V) #X times vectors from V by each mode
```
The *n-rank* and the *mutlilinear rank* of a tensor in Tucker format:
```julia
X=randttensor([9,8,7],[5,4,3]);
n=2;
nrank(X,n)
mrank(X)
```
HOSVD of a tensor in Tucker format:
```julia
X=randttensor([6,7,5],[4,4,4]);
hosvd(X)  #same as hosvd(X,eps_abs=1e-8)
hosvd(X,eps_abs=1e-6) #discard singular values lower than 1e-5
hosvd(X,eps_rel=1e-3) #discard singular values lower than 1e-3*sigma_{max}
hosvd(X,reqrank=[3,3,3]) #HOSVD with predefined multilinear rank
```
