using Tensors, Gadfly, JLD, DataFrames, TimeIt, Compose, Colors

@doc """ Function for testing Variants A and B of fast matrix-vector multiplication, when matrix is exactly matricized
Hadamard product of two randomly generated tensors in Tucker format of size IxIxI and multilinear ranks (R,R,R)
and v is a random vector.""" ->
function matvectest{D<:Integer}(I::Integer,ranks::Vector{D},n::Integer;create_plot=0,dir=".")
#Required:
#I ... defines size of a tensor as IxIxI
#ranks ... array of multilinear ranks [R1,R2,...,Rk]; creates tensors with multilinear ranks (Ri,Ri,Ri), i=1,...k
#n ... mode for matriciaztion of Hadamard product of tensors
#Optional:
#create_plot, dir ... set $create_plot to 1 if you want to create plot that will be saved in the directory $dir
#                     (make sure the directory exists!)
#
#Example: matvectest(1000,10:5:50,1,create_plot=1,dir="plots/")


  #I=1000;
  #ranks=[10:5:50]
  #n=1 #n-mode matricization

  v=rand(I);

  timeA=zeros(length(ranks));
  timeB=zeros(length(ranks));
  i=1;

  println("\n\n**Testing matvec.")

  for R in ranks
    println("\nTwo random tensors of size $I and rank $R.")
    T1=randttensor([I,I,I],[R,R,R]);
    T2=randttensor([I,I,I],[R,R,R]);
    println("Variant A:")
    timeA[i] = @timeit mhadtv(T1,T2,mhadtv(T1,T2,v,1,'t'),n,'n');
    println("Variant B:")
    timeB[i] = @timeit mhadtv(T1,T2,v,n);
    i+=1;
  end

  if create_plot == 1
    println("\n\n**Creating plot.")

    L=length(ranks);
    rnks=zeros(2*L); [rnks[2*k-1]=ranks[k] for k=1:L];[rnks[2*k]=ranks[k] for k=1:L];
    times=zeros(2*L); [times[2*k-1]=timeA[k] for k=1:L];[times[2*k]=timeB[k] for k=1:L];
    variants=Array(Any,2*L); [variants[2*k-1]="Variant A" for k=1:L];[variants[2*k]="Variant B" for k=1:L];

    df=DataFrame();
    df[:Rank]=rnks;
    df[:Time]=times;
    df[:Variant]=variants;

    p=plot(df,x="Rank",y="Time",color="Variant", Stat.xticks(ticks=df[:Rank]),
       Geom.line, Guide.xlabel("R"),Guide.ylabel("Time (s)"),Guide.Title("I=$I"));
    draw(SVG("$(dir)/matvec$I.svg",15cm,8cm),p)
  end
end
matvectest{D<:Integer}(I::Integer,ranks::Range{D},n::Integer;create_plot=0,dir=".")=matvectest(I,collect(ranks),n,create_plot=create_plot,dir=dir)

@doc """ Function for calculating error of HOSVD3 and HOSVD4 algorithms with required multilinear rank (R,R,R)
versus oversampling parameter p for predefined function related tensors.""" ->
function errortest{D<:Integer}(R::Integer,osp::Vector{D};f=1,create_plot=0,dir=".")
#Required:
#R ... required multlinear rank (R,R,R)
#osp ... vector that contains oversampling parameters
#Optional:
#f ... element of {1,2,3,4} for predefined function-related tensors of size 50x50x50 (1), 100x100x100 (2),
#           200x200x200 (3), 400x400x400 (4)
#create_plot, dir ... set $create_plot to 1 if you want to create plot that will be saved in the directory $dir
#                     (make sure the directory exists!)
#
#Example: errortest(9,[0,1,2,10],create_plot=1,dir="plots/")

  #R=9 #R=9,15
  #osp=[0,1,2,10]

  L=length(osp);

  println("\n\n**Initializing tensor.")
  T1=load("test_data_func.jld","F$(2*f-1)");
  T2=load("test_data_func.jld","F$(2*f)")
  println("\nTucker tensors of size ",size(T1)," based on functions.\n")
  Z=full(T1).*full(T2);
  sz=size(T1)[1];
  ranks=coresize(T1)[1];

  println("\n\n**Runing HOSVD algorithms.")

  algs=[3,4]
  for method in algs

    println("\nMethod hosvd$method:");

    err=Symbol("err_$(method)");
    @eval ($(err)=zeros($L));

    k=1;
    #println("\nTwo function-related tensors of size $(sz[i]).");
    for p in osp
      println("\nOversampling parameter : p = $p.")
        if method == 3
	        println("T=hosvd3(T1,T2,reqrank=[$R,$R,$R],eps_abs=1e-30,p=$p) ...")
          T=hosvd3(T1,T2,variant='A',reqrank=[R,R,R],eps_abs=1e-30,p=p);
        elseif method == 4
          println("T=hosvd4(T1,T2,reqrank=[$R,$R,$R],tol=1e-16,p=$p) ...")
          T=hosvd4(T1,T2,reqrank=[R,R,R],tol=1e-16,p=p);
        end
      @eval ($err[$k]=norm(full($T)-$Z));
      println("err[$k] = ",@eval ($err[$k]));
      k+=1;
    end
  end

  if create_plot == 1

    println("\n\n**Creating plot.")
    method_names=["HOSVD1","HOSVD2","HOSVD3","HOSVD4"][algs];

    nmbr=length(algs);
    errs=zeros(nmbr*L);
    mthds=Array(Any,nmbr*L);
    ops=zeros(Int,nmbr*L);

    i=1;
    for n=nmbr-1:-1:0
      err_Symbol=Symbol("err_$(algs[i])");
      [errs[nmbr*k-n]=(@eval $(err_Symbol)[$k]) for k=1:L];
      [mthds[nmbr*k-n]=method_names[i] for k=1:L];
      [ops[nmbr*k-n]=osp[k] for k=1:L];
      i+=1;
    end

    df=DataFrame();
    df[:OP]=ops;
    df[:Method]=mthds;
    df[:Error]=errs;

    #manual colors:
    a=Scale.color_discrete_hue(); a.f(4);

    p=plot(df, xgroup="OP", x="Method", y="Error", color="Method",
                       Geom.subplot_grid(Geom.bar(position=:dodge),Guide.xticks(label=false)),
                       Scale.y_log10, Guide.xlabel(nothing),Guide.Title("R=$R"),Scale.color_discrete_manual(a.f(4)[3],a.f(4)[4]));
    draw(SVG("$(dir)/errorsR$(R).svg",9cm,9cm),p)

  end
end
errortest(R::Integer,osp::Integer;f=1,create_plot=0,dir=".")=errortest(R,[osp],f=f,create_plot=create_plot,dir=dir)

@doc """ Function for testing algorithms HOSVD1, HOSVD2, HOSVD3, HOSVD4 on Tucker tensors of order 3,
         size IxIxI and multilinear ranks (R,R,R).""" ->
function hosvdtest(tentype;ranks=0,algs=[1,2,3,4],create_plot=0,dir=".",plotymax=0)
#Required:
#tentype ... Type and size of tensors. Either:
#         - "rand$sz" for randomly generated tensors, where $sz is the size of the tensor, e.g. "rand50", "rand100"
#         - "func$nmb" for predefined function related tensors, where $nmb can be left empty for all four predefined tensors or
#             can be specified to include only some of predefined tensors, where 1 means tensor of size I=50,
#             2 means tensor of size I=100, 3 means tensor of size I=200 and 4 means tensor of size I=400,
#             e.g. "func12" will only include tensors of size 50 and 100, "func4" will only include tensor of size 400;
#             "func1234" is the same as "func"
#Optional:
#ranks ... Only for randomly generated tensor. We have predefined ranks for tensors of size 50, 100, 200, 400, 500, 900:
#           sz=50 -> ranks=2:2:20; sz=100 -> ranks=3:3:30; sz=200 -> ranks=4:4:40;
#           sz=400 -> ranks=5:5:50; sz=500 -> ranks=5:5:50; sz=900 -> ranks=10:10:90;
#           In these cases $ranks does not have to be specified (i.e. if you want to use default values). If using randomly generated
#           tensors of not predefined size,then it has to be specified, e.g. hosvdtest("rand75",ranks=2:2:4), will run two tests,
#           on tensors of size 75x75x75 and multilinear ranks 2x2x2 and on tensors of size 75x75x75 and multilinear ranks 4x4x4;
#           hosvdtest("rand50",ranks=2:2:20) is the same as hosvdtest("rand50")
#algs ... Algorithms HOSVD1, HOSVD2, HOSVD3, HOSVD4 defined by their numbers. If not set, all four algorithms will be tested,
#         e.g. hosvdtest("rand50",algs=[1,2]) will only test HOSVD1 and HOSVD2 algorithms.
#create_plot, dir ... Setting $create_plot to 1 will create plot from the results in the directory specified in $dir,
#                     e.g. hosvdtest("rand50", create_plot=1, dir="plots"). Make sure the directory exists!
#plotymax ... Upper limit for y axis when plotting, predfinied for predefined tesors,
#            e.g. hosvdtest("rand50", create_plot=1, dir="plots",plotymax=0.2) is the same as hosvdtest("rand50", create_plot=1, dir="plots")
#
#Example: hosvdtest("rand200", create_plot=1, dir="plots")
#         hosvdtest("func", create_plot=1, dir="plots")

  if tentype[1:4] == "rand"
    rand_example=1;func_example=0;
    sz=parse(Int,tentype[5:end]);
    if !(sz in [50,100,200,400,500,900]) && ranks == 0
      error("Rank(s) not defined!")
    end
  elseif tentype[1:4] == "func"
    func_example=1;rand_example=0;
    if length(tentype)>4
      funcs=reverse(digits(parse(Int,tentype[5:end])))
    end
    if rank!=0
      warn("Function related tensor have predefined ranks. Ignoring ranks = $ranks.")
    end
  end

  println("\n\n**Initializing tensors.")

  if rand_example == 1

    if sz==50 && ranks==0; ranks=2:2:20; plotymax=0.2; end
    if sz==100 && ranks==0; ranks=3:3:30; plotymax=2; end
    if sz==200 && ranks==0; ranks=4:4:40; plotymax=8; end
    if sz==400 && ranks==0;ranks=5:5:50; plotymax=80; end
    if sz==500 && ranks==0;ranks=5:5:50; plotymax=12; end
    if sz==900 && ranks==0;ranks=10:10:90; plotymax=16; end

    limit=500; #for using @timeit
    L=length(ranks);
    println("\nCreating random tensors of size $sz and ranks $ranks.");

    T1=Array{Any}(L);T2=Array{Any}(L);
    i=1;
    for R in ranks
      T1[i]=randttensor([sz,sz,sz],[R,R,R]);
      T2[i]=randttensor([sz,sz,sz],[R,R,R]);
      println("-> Tucker tensors of size $sz and rank $R.");
      i+=1;
    end
  end

  if func_example == 1

    L=length(funcs);
    sz=zeros(Int,L);
    ranks=zeros(Int,L);
    println("\nCreating function related tensors.");

    T1=Array{Any}(L);T2=Array{Any}(L);
    i=1;
    for f in funcs
      T1[i]=load("test_data_func.jld","F$(2*f-1)");
      T2[i]=load("test_data_func.jld","F$(2*f)")
      println("-> Tucker tensors of size ",size(T1[i])," based on functions.")
      sz[i]=size(T1[i])[1];
      ranks[i]=coresize(T1[i])[1];
      i+=1;
    end
  end


  println("\n\n**Runing HOSVD algorithms.")

  for method in algs

    println("\nMethod hosvd$method.");

    time=Symbol("time_$(method)");
    err=Symbol("err_$(method)");
    @eval ($(time)=zeros($L));
    @eval ($(err)=zeros($L));

    i=1;
    for R in ranks
      if rand_example == 1
        println("\nTwo random tensors of size $sz and rank $R.");
        if method==1
	        println("T=hosvd1(T1,T2,[$R,$R,$R]) ...")
          if sz >= limit && i>1
	          tic(); T=hosvd1(T1[i],T2[i],reqrank=[R,R,R],); @eval $(time)[$i]=toc();
	        else
            @eval ($(time)[$i]= @timeit T=hosvd1($(T1[i]),$(T2[i]),reqrank=[$R,$R,$R]));
          end
        elseif method==2
          println("T=hosvd2(T1,T2,[$R,$R,$R]) ...")
          if sz >= limit && i>1
            tic(); T=hosvd2(T1[i],T2[i],reqrank=[R,R,R]); @eval $(time)[$i]=toc();
	        else
            @eval ($(time)[$i]= @timeit T=hosvd2($(T1[i]),$(T2[i]),reqrank=[$R,$R,$R]));
          end
        elseif method == 3
          println("T=hosvd3(T1,T2,[$R,$R,$R]) ...")
	        if R^2>sz;var='A';else; var='B';end
	        if sz >= limit && i>1
		        tic(); T=hosvd3(T1[i],T2[i],reqrank=[R,R,R],variant=var); @eval $(time)[$i]=toc();
	        else
            @eval ($(time)[$i]= @timeit T=hosvd3($(T1[i]),$(T2[i]),reqrank=[$R,$R,$R],variant=$var));
          end
        elseif method==4
          println("T=hosvd4(T1,T2,[$R,$R,$R]) ...")
          if sz >= limit && i>1
            tic(); T=hosvd4(T1[i],T2[i],reqrank=[R,R,R]); @eval $(time)[$i]=toc();
	        else
            @eval ($(time)[$i]= @timeit T=hosvd4($(T1[i]),$(T2[i]),reqrank=[$R,$R,$R]));
          end
        end
      end
      if func_example == 1
        println("\nTwo function-related tensors of size $(sz[i]).");
        if method == 1
          println("T=hosvd1(T1,T2) ...")
          @eval ($(time)[$i]= @timeit T=hosvd1($(T1[i]),$(T2[i])));
        elseif method == 2
          println("T=hosvd2(T1,T2) ...")
          @eval ($(time)[$i]= @timeit T=hosvd2($(T1[i]),$(T2[i])));
        elseif method == 3
	        if i==4; var='B';else;var='A';end
          println("T=hosvd3(T1,T2) ...")
          @eval ($(time)[$i]= @timeit T=hosvd3($(T1[i]),$(T2[i]),variant=$var));
        elseif method == 4
          println("T=hosvd4(T1,T2) ...")
          @eval ($(time)[$i]= @timeit T=hosvd4($(T1[i]),$(T2[i])));
        end
      end
      i+=1;
    end
  end

  if create_plot == 1
    println("\n\n**Creating plot.")

    method_names=["HOSVD1","HOSVD2","HOSVD3","HOSVD4"][algs];
    nmbr=length(algs);
    rnks=zeros(nmbr*L);
    times=zeros(nmbr*L);
    mthds=Array(Any,nmbr*L);
    if func_example == 1;   szs=zeros(nmbr*L); end

    i=1;
    for n=nmbr-1:-1:0
      [rnks[nmbr*k-n]=ranks[k] for k=1:L];
      time_Symbol=Symbol("time_$(algs[i])");
      [times[nmbr*k-n]=(@eval $(time_Symbol)[$k]) for k=1:L];
      [mthds[nmbr*k-n]=method_names[i] for k=1:L];
      if func_example == 1; [szs[nmbr*k-n]=sz[k] for k=1:L]; end
      i+=1;
    end

    df=DataFrame();
    df[:Rank]=round(Int32,rnks);
    df[:Time]=times;
    df[:Method]=mthds;
    if func_example == 1; df[:Size]=round(Int64,szs); end

    #manual colors:
    #a=Scale.color_discrete_hue(); #a.f(4) first 4 colors

    if rand_example == 1
      if plotymax!=0
        p=plot(df,x="Rank",y="Time", color="Method", Stat.xticks(ticks=df[:Rank]),Geom.line,Coord.Cartesian(ymin=0,ymax=plotymax),
                 Guide.xlabel("R"),Guide.Title("I=$sz"));#,Scale.color_discrete_manual(a.f(4)[2],a.f(4)[3],a.f(4)[4]));
      else
        p=plot(df,x="Rank",y="Time", color="Method", Stat.xticks(ticks=df[:Rank]),Geom.line,Coord.Cartesian,
                 Guide.xlabel("R"),Guide.Title("I=$sz"));#,Scale.color_discrete_manual(a.f(4)[2],a.f(4)[3],a.f(4)[4]));
      end
      draw(SVG("$(dir)/time_rand$sz.svg",12cm,8cm),p)
    end

    if func_example==1
      i=1;
      if length(funcs) == 1
        p1=plot(df[nmbr*(i-1)+1:nmbr*i,1:end],x="Method",y="Time",color="Method",Geom.bar(position=:dodge),
                Guide.xticks(label=false),Guide.xlabel(nothing),Guide.Title("I=$(sz[i])"))
        draw(SVG("$(dir)/time_func.svg",3.5cm,8cm),plt);
      else
        p1=plot(df[nmbr*(i-1)+1:nmbr*i,1:end],x="Method",y="Time",color="Method",Geom.bar(position=:dodge),
                Guide.xticks(label=false),Guide.xlabel(nothing),Guide.Title("I=$(sz[i])"),Theme(key_position = :none));
        i+=1;
        if length(funcs) == 2
          p2=plot(df[nmbr*(i-1)+1:nmbr*i,1:end],x="Method",y="Time",color="Method",Geom.bar(position=:dodge),
                  Guide.xticks(label=false), Guide.xlabel(nothing),Guide.ylabel(nothing),Guide.Title("I=$(sz[i])"));
          plt=hstack(compose(context(0, 0, 3.5cm, 8cm), render(p1)),
                    compose(context(0, 0, 3.9cm, 8cm), render(p2)))
          draw(SVG("$(dir)/time_func.svg",8cm,8cm),plt);
        else
          p2=plot(df[nmbr*(i-1)+1:nmbr*i,1:end],x="Method",y="Time",color="Method",Geom.bar(position=:dodge),
                  Guide.xticks(label=false),Guide.xlabel(nothing),Guide.ylabel(nothing),Guide.Title("I=$(sz[i])"),Theme(key_position = :none));
          i+=1;
          if length(funcs) == 3
            p3=plot(df[nmbr*(i-1)+1:nmbr*i,1:end],x="Method",y="Time",color="Method",Geom.bar(position=:dodge),
                    Guide.xticks(label=false),Guide.xlabel(nothing),Guide.ylabel(nothing),Guide.Title("I=$(sz[i])"));
            plt=hstack(compose(context(0, 0, 3.5cm, 8cm), render(p1)),
                  compose(context(0, 0, 2.7cm, 8cm), render(p2)),
                  compose(context(0, 0, 3.9cm, 8cm), render(p3)))
            draw(SVG("$(dir)/time_func.svg",11cm,8cm),plt);
          else
            p3=plot(df[nmbr*(i-1)+1:nmbr*i,1:end],x="Method",y="Time",color="Method",Geom.bar(position=:dodge),
                    Guide.xticks(label=false),Guide.xlabel(nothing),Guide.ylabel(nothing),Guide.Title("I=$(sz[i])"),Theme(key_position = :none));
            i+=1
            p4=plot(df[nmbr*(i-1)+1:nmbr*i,1:end],x="Method",y="Time",color="Method",Geom.bar(position=:dodge),
                    Guide.xticks(label=false), Guide.xlabel(nothing),Guide.ylabel(nothing),Guide.Title("I=$(sz[i])"));
            plt=hstack(compose(context(0, 0, 3.5cm, 8cm), render(p1)),
              compose(context(0, 0, 2.7cm, 8cm), render(p2)),
              compose(context(0, 0, 2.5cm, 8cm), render(p3)),
              compose(context(0, 0, 3.9cm, 8cm), render(p4)));
            draw(SVG("$(dir)/time_func.svg",15cm,8cm),plt);
          end
        end
      end
    end
  end
end

@doc """ Function for testing algorithm HOSVD4 on Tucker tensors of order 3,
         with fixed multilinear rank (R,R,R) and different sizes IxIxI.""" ->
function hosvd4test{D<:Integer}(R,I::Vector{D};create_plot=0,dir=".")
#Required:
#R ... multilinear rank (R,R,R)
#I ... vector of sizes [I1,I2,...,In]
#Optional:
#create_plot, dir ... set $create_plot to 1 if you want to create plot that will be saved in the directory $dir
#                     (make sure the directory exists!)
#
#Example: hosvd4test(50,500:500:5000,create_plot=1,dir="plots/")

  limit=500; #for using @timeit

  println("\n\n**Initializing tensors.")

  L=length(I);
  T1=Array{Any}(L);T2=Array{Any}(L);
  i=1;
  for sz in I
    println("\nCreating random tensors of size $sz and ranks $R.");
    T1[i]=randttensor([sz,sz,sz],[R,R,R]);
    T2[i]=randttensor([sz,sz,sz],[R,R,R]);
    i+=1;
  end

  println("\n\n**Runing HOSVD4 algorithm.")

  times=zeros(L);
  i=1;
  for sz in I
    println("T=hosvd4(T1,T2,[$R,$R,$R]) ...")
    if sz >= limit && i>1
      tic(); T=hosvd4(T1[i],T2[i],reqrank=[R,R,R]); times[i]=toc();
	  else
      times[i]= @timeit T=hosvd4(T1[i],T2[i],reqrank=[R,R,R]);
    end
    i+=1
  end

  if create_plot == 1
    println("\n\n**Creating plot.")


    df=DataFrame();
    df[:Size]=I;
    df[:Time]=times;
    df[:Method]=["HOSVD4" for i=1:L];

    #manual colors:
    a=Scale.color_discrete_hue(); a.f(4)

    p=plot(df,x="Size",y="Time", color="Method", Stat.xticks(ticks=df[:Size]),Geom.line,Coord.Cartesian,
                 Guide.xlabel("I"),Guide.Title("R=$R"),Scale.color_discrete_manual(a.f(4)[4]));
    draw(SVG("$(dir)/hosvd4_time_rand$R.svg",12cm,8cm),p)
  end

end
hosvd4test{D<:Integer}(R,I::Range{D};create_plot=0,dir=".")=hosvd4test(R,collect(I),create_plot=create_plot,dir=dir)
#
