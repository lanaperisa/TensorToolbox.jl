println("\n")

sp = sptenrand([10, 10, 10], 0.01)
dense = Array(sp)

println("... Testing sparse and dense tensor product equivalence.")
@test ttt(sp,sp) == ttt(dense,dense)
