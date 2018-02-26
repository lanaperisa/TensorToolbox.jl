#using TensorToolbox
#using Base.Test

println("\n\n****Testing dimtree.jl")

println("\n...Testing creating a dimtree.")
T1=dimtree([8,9,5,6,7],[1,2,3,4])
T2=dimtree([5,6,7,8,9],[1,2,3,4])
@test T1==T2
T3=dimtree(5,internal_nodes=[1,2,3,4])
@test T1==T3
C=children(T1)
T4=dimtree(C)
@test T3==T4
T5=dimtree(5,leaves=collect(5:9))
@test T3==T5

T1=dimtree(5,internal_nodes=[1,2,3,4])
T2=dimtree(5,leaves=[2,4,6,8,9])
@test non(T1)==non(T2)

println("\n...Created two dimtrees for tensor of order 5, with ",non(T1)," nodes.")

println("\n...Testing functions:")
println(" - children")
c=children(T1,2)
@test c==[4,5]
c=children(T2,3)
@test c==[4,5]
println(" - sibling")
@test sibling(T1,4)==5
@test sibling(T2,5)==4
println(" - parent")
p=parent(T1,4)
@test p==2
p=parent(T2,5)
@test p==3
println(" - is_left")
@test is_left(T1,6) == true
println(" - is_right")
@test is_right(T1,5) == true
println(" - count_leaves")
l=count_leaves(T1,2)
@test l==3
l=count_leaves(T2,5)
@test l==3
println(" - dims")
@test dims(T1,4)==[1,2]
@test dims(T2,3)==[2,3,4,5]
println(" - lvl")
@test lvl(T1,9)==3
@test lvl(T2,5)==2
println(" - nodes_on_lvl")
@test nodes_on_lvl(T1,2)==[4,5,6,7]
@test nodes_on_lvl(T2,3)==[6,7]
println(" - subnodes")
@test subnodes(T2,5)==[5,6,7,8,9]
