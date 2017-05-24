using Tensors,JLD


f1(x,y,z)=1/(x+y+z);
f2(x,y,z)=1/((x+y+z)^(1/2));

X1=cell(4);X2=cell(4);
sz=[50,100,200,400]
i=1;
for s in sz
  dx=dy=dz=[n*0.1 for n=1:s];
  F1=Float64[ f1(x,y,z) for x=dx, y=dy, z=dz ];
  F2=Float64[ f2(x,y,z) for x=dx, y=dy, z=dz ];
  if i==1
    eps1=1e-7;
    eps2=1e-8;
  elseif i==2
    eps1=1e-7;
    eps2=1e-7;
  else
    eps1=1e-8;
    eps2=1e-8;
  end

  X1[i]=hosvd(F1,eps_abs=eps1);
  X2[i]=hosvd(F2,eps_abs=eps2);

  println("\n\nTucker tensors of size $s based on functions.\n")
  println("X1 is a tensor based on function f1(x)=1/(x+y+z) and of rank ",coresize(X1[i]),".")
  println("X2 is a tensor based on function f2(x)=1/((x+y+z)^(1/2)) and of rank ",coresize(X2[i]),".")

  Z=full(X1[i]).*full(X2[i]);
  T=hosvd(Z)
  println("\nHadamard product Z=X1.*X2 has multilinear rank ",coresize(T));

  i+=1;

end
save("test_data_func.jld","F1",X1[1],"F2",X2[1],"F3",X1[2],"F4",X2[2],"F5",X1[3],"F6",X2[3],"F7",X1[4],"F8",X2[4]);

