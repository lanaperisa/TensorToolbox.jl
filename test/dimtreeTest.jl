using TensorToolbox, Test, LinearAlgebra

println("\n\n****Testing dimtree.jl")

println("\n...Testing creating a dimtree.")
T1=dimtree([8,9,5,6,7],[1,2,3,4])
T2=dimtree([8,9,5,6,7])
@test T1==T2
T3=dimtree([5,6,7,8,9],[1,2,3,4])
T4=dimtree(collect(5:9))
@test T3==T4

println("...Testing function non (number-of-nodes).")
@test non(T1)==non(T3)

println("\n...Created two dimtrees for tensor of order 5, with ",non(T1)," nodes.")

println("\n...Testing functions:")
println(" - height")
h=height(T1)
@test h==4
println(" - children")
c=children(T1,2)
@test c==[4,5]
c=children(T3,3)
@test c==[6,7]
println(" - sibling")
@test sibling(T1,4)==5
@test sibling(T2,5)==4
println(" - parent")
p=parent(T1,4)
@test p==2
p=parent(T3,6)
@test p==3
println(" - is_leaf")
@test is_leaf(T1,5) == true
println(" - is_left")
@test is_left(T1,6) == true
println(" - is_right")
@test is_right(T1,5) == true
println(" - count_leaves")
l=count_leaves(T1,2)
@test l==3
l=count_leaves(T3,3)
@test l==2
println(" - dims")
@test dims(T1,4)==[1,2]
@test dims(T3,3)==[2,3]
println(" - lvl")
@test lvl(T1,9)==3
@test lvl(T2,5)==2
println(" - nodes_on_lvl")
@test nodes_on_lvl(T1,2)==[4,5,6,7]
@test nodes_on_lvl(T3,3)==[8,9]
println(" - node2ind")
@test node2ind(T1,6)==4
@test node2ind(T3,[3,5])==[3,1]
println(" - subnodes")
@test subnodes(T3,4)==[4,8,9]
println(" - left_child_length")
@test left_child_length(T1,2)==2
