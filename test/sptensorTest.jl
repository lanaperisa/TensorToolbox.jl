import Random
Random.seed!(1337)

println("\n")

sp = sptenrand([10, 10, 10], 0.01)
dense = Array(sp)

println("... Testing sparse and dense tensor product equivalence.")
@test ttt(sp,sp) == ttt(dense,dense)

println("... Testing sparse tensor decomposition")
@test cp_als(reduce(ttt,SparseTensor.([[0, 0, 1],[0, 1, 0],[1, 0, 0]])),1) == ktensor([[0 0 1],[0 1 0],[1 0 0]].|>permutedims .|> x->x .|> Float64)
