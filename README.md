# TensorToolbox.jl

[![Build Status](https://travis-ci.org/lanaperisa/TensorToolbox.jl.svg?branch=master)](https://travis-ci.org/lanaperisa/TensorToolbox.jl)

Julia package for tensors. Includes functionality for
- dense tensors, 
- tensors in Tucker format,
- tensors in Kruskal (CP) format,
- tensors in Hierarchical Tucker format,
- tensors in Tensor Train format (work in progress).

Follows the functionality of MATLAB [Tensor toolbox](http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html) and [Hierarchical Tucker Toolbox](https://anchp.epfl.ch/htucker).

Additionally, it contains algorithms from the paper [Recompression of Hadamard Products
of Tensors in Tucker Format](http://sma.epfl.ch/~anchpcommon/publications/ttensors_pp.pdf) by D. Kressner and L. Periša.

## Basics

Start with
```julia
using TensorToolbox
```

Define tensor as multidimensional array and calculate its norm:
```julia
X=rand(4,3,2)
norm(X)
```
Create identity and diagonal tensor:
```julia
Id=neye(2,2,2)
D=diagt([1,2,3,4])
```
For two tensors of same size calculate their inner product:
```julia
X=rand(3,3,3,3);Y=rand(3,3,3,3);
innerprod(X,Y)
```
*Matricization* of a tensor:
```julia
X=rand(4,3,2);n=1; 
A=tenmat(X,n) #by mode n
B=tenmat(X,row=[2,1],col=3) #by row modes [2,1] and column mode 3
```
Fold matrix back to tensor:
```julia
X=matten(A,n,[4,3,2]) # by mode n
X=matten(B,[2,1],[3],[4,3,2]) # by row modes [2,1] and column mode 3
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
hosvd(X) #same as hosvd(X,eps_abs=1e-8)
hosvd(X,eps_abs=1e-6) #discard singular values lower than 1e-5
hosvd(X,eps_rel=1e-3) #discard singular values lower than 1e-3*sigma_{max}
hosvd(X,reqrank=[2,2,2])
```
The CP decomposition:
```julia
X=rand(5,4,3);
R=3; #number of components
cp_als(X,R)  #same as cp_als(X,R,init="rand",dimorder=1:ndims(X))
cp_als(X,R,init=[rand(5,3),rand(4,3),rand(3,3)]) #initialize factor matrices 
cp_als(X,R,init="nvecs",dimorder=[2,1,3])
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
The CP decomposition:
```julia
X=randttensor([6,7,5],[4,4,4]);
R=3; #number of components
cp_als(X,R)  #same as cp_als(X,R,init="rand",dimorder=1:ndims(X))
cp_als(X,R,init=[rand(6,3),rand(7,3),rand(5,3)]) #initialize factor matrices 
cp_als(X,R,init="nvecs",dimorder=[2,1,3])
```

## Tensors in Kruskal format

Define tensor in Kruskal format by its factor matrices (and vector of weights):
```julia
lambda=rand(3)
A=[rand(5,3),rand(4,3),rand(3,3)];
ktensor(A)
ktensor(lambda,A)
```
Create random tensor in Kruskal format of size 5x4x3 and with 2 components: 
```julia
X=randktensor([5,4,3],2)
```
Basic functions:
```julia
size(X) 
ndims(X)
norm(X)
full(X)  #Creates full tensor out of Kruskal format
permutedims(X,[2,1,3]) 
ncomponents(X) #Number of components
```
*n-mode matricization* of a tensor in Kruskal format:
```julia
n=1;
tenmat(X,n)
```
Basic operations:
```julia
X=randktensor([5,4,3],2);Y=randktensor([5,4,3],3);
innerprod(X,Y)
X+Y
X-Y
X==Y #same as isequal(X,Y)
3*X #same as mtimes(3,X)
```
*n-mode product* of a tensor in Kruskal format and a matrix or an array of matrices:
```julia
X=randktensor([5,4,3],2);
A=[rand(2,5),rand(2,4),rand(2,3)];
ttm(X,A[1],1)  #X times A[1] by mode 1
ttm(X,[A[1],A[2]],[1,2]) #X times A[1] by mode 1 and times A[2] by mode 2; same as ttm(X,A,-3)
ttm(X,A) #X times matrices from A by each mode
```
*n-mode (vector) product* of a tensor in Kruskal format and a vector or an array of vectors:
```julia
X=randktensor([5,4,3],2);
V=[rand(5),rand(4),rand(3)];
ttv(X,V[1],1)  #X times V[1] by mode 1
ttv(X,[V[1],V[2]],[1,2]) #X times V[1] by mode 1 and times V[2] by mode 2; same as ttm(X,V,-3)
ttv(X,V) #X times vectors from V by each mode
```
Arrange the rank-1 components of a tensor in Kruskal format:
```julia
X=randktensor([6,5,4,3],3);
arrange(X)
arrange!(X)
```
Fix sign ambiguity of a tensor in Kruskal format:
```julia
X=randktensor([6,5,4,3,4],3);
fixsigns(X)
fixsigns!(X)
```
Distribute weights a tensor in Kruskal format to a specific mode:
```julia
X=randktensor([3,3,3],3);
n=2;
redistribute(X,n)
redistribute!(X,n)
```
The CP decomposition:
```julia
X=randktensor([6,7,5],4);
R=3; #number of components
cp_als(X,R)  #same as cp_als(X,R,init="rand",dimorder=1:ndims(X))
cp_als(X,R,init=[rand(6,3),rand(7,3),rand(5,3)]) #initialize factor matrices 
cp_als(X,R,init="nvecs",dimorder=[2,1,3])
```
## Tensors in Hierarchical Tucker format

Define tensor in Hierarchical Tucker format by dimensional tree T, its transfer tensors and factor matrices:
```julia
T=dimtree(3)
B=[rand(2,3,1),rand(4,3,2)]
A=[rand(5,4),rand(4,3),rand(3,3)]
htensor(T,B,A)
```
Define tensor in Hierarchical Tucker format by dimensional tree T, its transfer tensors and factor matrices:
```julia
T=dimtree(3)
B=[rand(2,3,1),rand(4,3,2)]
A=[rand(5,4),rand(4,3),rand(3,3)]
htensor(T,B,A)
```
Get Tucker format of a tensor by using htrunc:
```julia
X=rand(8,9,7);
htrunc(X)
htrunc(X,maxrank=3) #hrunc with defined maximal rank
```
Create random tensor in Hierarchical Tucker format of size 5x4x3:
```julia
X=randhtensor([5,4,3])
```
Basic functions:
```julia
size(X)
ndims(X)
norm(X)
full(X)  #Creates full tensor out of Hierarchial Tucker format
reorth(X) #Orthogonalize factor matrices
```
Basic operations:
```julia
X=randhtensor([5,4,3]);Y=randhtensor([5,4,3]);
innerprod(X,Y)
X+Y
X-Y
X==Y #same as isequal(X,Y)
3*X #same as mtimes(3,X)
```
*n-mode product* of a tensor in Hierarchical Tucker format and a matrix or an array of matrices:
```julia
X=randhtensor([5,4,3]);
A=[rand(2,5),rand(2,4),rand(2,3)];
ttm(X,A[1],1)  #X times A[1] by mode 1
ttm(X,[A[1],A[2]],[1,2]) #X times A[1] by mode 1 and times A[2] by mode 2; same as ttm(X,A,-3)
ttm(X,A) #X times matrices from A by each mode
```
*n-mode (vector) product* of a tensor in Hierarchical Tucker format and a vector or an array of vectors:
```julia
X=randhtensor([5,4,3]);
V=[rand(5),rand(4),rand(3)];
ttv(X,V[1],1)  #X times V[1] by mode 1
ttv(X,[V[1],V[2]],[1,2]) #X times V[1] by mode 1 and times V[2] by mode 2; same as ttm(X,V,-3)
ttv(X,V) #X times vectors from V by each mode
```
The *h-rank* of a tensor in Hierarchical Tucker format:
```julia
X=htrunc(rand(9,8,7),maxrank=2)
hrank(X)
```

## Tensors in Tensor Train format

Define tensor in TT format by its core tensors:
```julia
G=CoreCell(undef,3)
G[1]=rand(1,4,3)
G[2]=rand(3,6,4)
G[3]=rand(4,3,1)
X=TTtensor(G)
```
Get TT format of a tensor by using TTsvd:
```julia
X=rand(5,4,3,2)
TTsvd(X)
TTsvd(X,reqrank=[2,2,2])
```
Create random TT tensor of size 5x4x3 and TT-rank (2,2):
```julia
X=randTTtensor([5,4,3],[2,2])
```
Basic functions::
```julia
size(X)
TTrank(X)
ndims(X)
norm(X)
full(X)  #Creates full tensor out of Tucker format
reorth(X)
```
Basic operations:
```julia
X=randTTtensor([5,4,3],[2,2])
Y=randTTtensor([5,4,3],[3,3])

innerprod(X,Y)
X+Y
X-Y
3*X
```
TTsvd of a TT tensor:
```julia
X=randTTtensor([7,6,5],[5,4])
TTsvd(X,reqrank=[3,3])
```
