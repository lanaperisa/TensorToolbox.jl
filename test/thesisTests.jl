using TensorToolbox, Gadfly, JLD, TimeIt, Compose, Colors

@doc """ Function for testing Variants A and B of the fast matrix-vector multiplication, when matrix is exactly matricized
Hadamard product of two randomly generated tensors in Tucker format of size SxSxS and multilinear ranks (R,R,R)
and v is a random vector.""" ->
function matvectest{D<:Integer}(S::Integer,ranks::Vector{D},n::Integer;create_plot=0,dir=".")
#Required:
#S ... defines size of a tensor as SxSxS
#ranks ... array of multilinear ranks [R1,R2,...,Rk]; creates k tensors of multilinear ranks (Ri,Ri,Ri), i=1,...,k
#n ... mode for matricization of Hadamard product of tensors
#Optional:
#create_plot, dir ... set $create_plot to 1 if you want to create plot that will be saved in the directory $dir
#                     (make sure the directory exists!)
#
#Example: matvectest(1000,10:5:50,1,create_plot=1,dir="plots/")


  #S=1000;
  #ranks=[10:5:50]
  #n=1 #n-mode matricization

  v=rand(S);

  timeA=zeros(length(ranks));
  timeB=zeros(length(ranks));
  i=1;

  println("\n\n**Testing matvec.")

  for R in ranks
    println("\nTwo random tensors of size $S and rank $R.")
    T1=randttensor([S,S,S],[R,R,R]);
    T2=randttensor([S,S,S],[R,R,R]);
    println("Variant A:")
    timeA[i] = @timeit mhadtv(T1,T2,mhadtv(T1,T2,v,1,'t'),n,'n');
    println("Variant B:")
    timeB[i] = @timeit mhadtv(T1,T2,v,n);
    i+=1;
  end

  if create_plot == 1
    println("\n\n**Creating plot.")

    a=Scale.color_discrete_hue();
    dash = 4 * Compose.mm

    p=plot(layer(x=ranks,y=timeA,Geom.line,Theme(default_color=color(a.f(2)[1]))),
           layer(x=ranks,y=timeB,Geom.line,Theme(default_color=color(a.f(2)[2]),line_style=[dash])),
          Stat.xticks(ticks=collect(ranks)),#Scale.y_log10,
          Guide.manual_color_key("Variant", ["\U2015 Variant A", "\U2010\U2010\U2010 Variant B"],[a.f(2)[1],a.f(2)[2]]),
          Guide.ylabel("Time (s)"),Guide.xlabel("R"),Guide.Title("I=$S"));
    draw(SVG("$(dir)/matvec$S.svg",15cm,8cm),p)
  end
end
matvectest{D<:Integer}(S::Integer,ranks::Range{D},n::Integer;create_plot=0,dir=".")=matvectest(S,collect(ranks),n,create_plot=create_plot,dir=dir)


@doc """ Function for calculating errors of HOSVD1, HOSVD2, HOSVD3 and HOSVD4 algorithms with required multilinear rank (R,R,R)
versus oversampling parameter p for predefined function related tensors.""" ->
function errortest{D<:Integer}(R::Integer,osp::Vector{D};f=1,create_plot=0,dir=".")
#Required:
#R ... required multlinear rank (R,R,R)
#osp ... vector that contains oversampling parameters [p1,...,pk]
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

  algs=[1,2,3,4]
  nmbr=length(algs);
  for method in algs

    println("\nMethod hosvd$method:");

    err=Symbol("err_$(method)");
    @eval ($(err)=zeros($L));

    k=1;
    #println("\nTwo function-related tensors of size $(sz[i]).");
    for p in osp
      println("\nOversampling parameter : p = $p.")
      if method == 1
	      println("T=hosvd1(T1,T2,reqrank=[$R,$R,$R],eps_abs=1e-30,p=$p) ...")
        T=hosvd1(T1,T2,reqrank=[R,R,R],eps_abs=1e-30,p=p);
      elseif method == 2
	      println("T=hosvd2(T1,T2,reqrank=[$R,$R,$R],eps_abs=1e-30,p=$p) ...")
        T=hosvd2(T1,T2,reqrank=[R,R,R],eps_abs=1e-30,p=p);
      elseif method == 3
	      println("T=hosvd3(T1,T2,reqrank=[$R,$R,$R],eps_abs=1e-30,p=$p) ...")
        T=hosvd3(T1,T2,variant='A',reqrank=[R,R,R],eps_abs=1e-30,p=p);
      elseif method == 4
        println("T=hosvd4(T1,T2,reqrank=[$R,$R,$R],tol=1e-16,p=$p) ...")
        T=hosvd4(T1,T2,reqrank=[R,R,R],tol=1e-16,p=p);
      end
      @eval ($err[$k]=vecnorm(full($T)-$Z));
      println("err[$k] = ",@eval ($err[$k]));
      k+=1;
    end
  end

  if create_plot == 1

    println("\n\n**Creating plot.")

    c=Scale.color_discrete_hue() #manual colors
    errors=zeros(nmbr*L)
    k=1;
    for i=1:L
      if 1 in algs
        errors[k]=(@eval $(Symbol("err_1")))[i]; k+=1;
      end
      if 2 in algs
        errors[k]=(@eval $(Symbol("err_2")))[i]; k+=1;
      end
      if 3 in algs
        errors[k]=(@eval $(Symbol("err_3")))[i]; k+=1;
      end
      if 4 in algs
        errors[k]=(@eval $(Symbol("err_4")))[i]; k+=1;
      end
    end

    op=sort(vec(repmat(osp,nmbr,1)));
    methods=vec(repmat(algs,L,1));
    colors=vec(repmat(["HOSVD1","HOVSD2","HOSVD3","HOSVD4"][algs],L,1))

      p=plot(xgroup=op, x=methods, y=errors, color=colors,
          Guide.colorkey("Method"),
          Geom.subplot_grid(Geom.bar(position=:dodge)),#Guide.xticks(label=false)),
          Scale.y_log10, Guide.xlabel(nothing),Guide.ylabel("Error"),Guide.Title("R=$R"),
 	        Scale.x_discrete);
    draw(SVG("$(dir)/errorsR$(R).svg",9cm,9cm),p)

  end
end
errortest(R::Integer,osp::Integer;f=1,create_plot=0,dir=".")=errortest(R,[osp],f=f,create_plot=create_plot,dir=dir)


@doc """ Function for testing algorithms HOSVD1, HOSVD2, HOSVD3, HOSVD4 on randomly generated Tucker tensors of order N,
         size Sx...xS and multilinear ranks (R,...,R).""" ->
function hosvdrandtest(S,ranks,N;algs=[1,2,3,4],timeit_limit=500,hosvd2_limit=0,create_plot=0,dir=".")
#Required:
#S ... defines size of a tensors as Sx..xS
#ranks ... array of multilinear ranks [R1,R2,...,Rk]; creates tensors of multilinear ranks (Ri,...,Ri), i=1,...,k
#N ... order of tensors
#Optional:
#algs ... Algorithms HOSVD1, HOSVD2, HOSVD3, HOSVD4 defined by their numbers. If not set, all four algorithms will be tested,
#         e.g. hosvdtest(50,[2,4,6],algs=[1,2]) will only test HOSVD1 and HOSVD2 algorithms.
#timeit_limit ... Replaces @timeit function, which runs function in a loop, with @time, which runs function only one, if S>timeit_limit.
#hosvd2_limit ... Since in some examples HOSVD2 algorithm works only up to certain R, set hosvd2_limit to skip HOSVD2 when R>hosvd2_limit.
#create_plot, dir ... Setting $create_plot to 1 will create plot from the results in the directory specified in $dir,
#                     e.g. hosvdtest(50,[2,4,6], create_plot=1, dir="plots"). Make sure the directory exists!
#
#Example: hosvdrandtest(50,collect(2:2:20),3,create_plot=1, dir="plots")

  println("\n\n**Initializing tensors.")

  L=length(ranks);
  println("\nCreating random tensors of size $S and ranks $ranks.");

  T1=Array{Any}(L);T2=Array{Any}(L);
  i=1;
  for R in ranks
    T1[i]=randttensor(S,R,N);
    T2[i]=randttensor(S,R,N);
    println("-> Tucker tensors of size $S and rank $R.");
    i+=1;
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
      println("\nTwo random tensors of order $N, size $S and rank $R.");
      if method==1
        println("T=hosvd1(T1,T2,$R) ...")
        if S >= timeit_limit && i>1
	        tic(); T=hosvd1(T1[i],T2[i],reqrank=R,); @eval $(time)[i]=toc();
	      else
          @eval ($(time)[$i]= @timeit T=hosvd1($T1[$i],$T2[$i],reqrank=$R));
      end
      elseif method==2
	      if R > hosvd2_limit
	 	     continue;
	      end
        println("T=hosvd2(T1,T2,$R) ...")
        if S >= timeit_limit && i>1
          tic(); T=hosvd2(T1[i],T2[i],reqrank=R); @eval $(time)[i]=toc();
	      else
          @eval ($(time)[$i]= @timeit T=hosvd2($T1[$i],$T2[$i],reqrank=$R));
      end
      elseif method == 3
        println("T=hosvd3(T1,T2,$R) ...")
	      if R^2>S;var='A';else; var='B';end
	      if S >= timeit_limit && i>1
		      tic(); T=hosvd3(T1[i],T2[i],reqrank=R,variant=var); @eval $(time)[i]=toc();
	      else
          @eval ($(time)[$i]= @timeit T=hosvd3($T1[$i],$T2[$i],reqrank=$R,variant=$var));
      end
      elseif method==4
        println("T=hosvd4(T1,T2,$R) ...")
        if S >= timeit_limit && i>1
          tic(); T=hosvd4(T1[i],T2[i],reqrank=R); @eval $(time)[i]=toc();
	      else
          @eval ($(time)[$i]= @timeit T=hosvd4($T1[$i],$T2[$i],reqrank=$R));
        end
      end
      i+=1;
    end
  end

  if create_plot == 1
    println("\n\n**Creating plot.")

    #manual colors:
    c=Scale.color_discrete_hue() #c.f(4);
    #for black&white:
    dash = 4 * Compose.mm;
    ddot = 0.5 * Compose.mm;
    gap = 1 * Compose.mm;
    lines=[[],[dash],[ddot],[dash,gap,ddot,gap]]
    legend=["\U2E3B HOSVD1",
          "\U2013\U2013\U2013\U2006 HOSVD2",
          "\U2010\U2010\U2010\U2010\U2010\U2010\U2010\U2006 HOSVD3",
          "\U2013 \U2010 \U2013 HOSVD4"]

    p=plot([layer(x=ranks,y=(@eval $(Symbol("time_$a"))),Geom.line,
                  Theme(default_color=color(c.f(4)[a]),line_style=lines[a])) for a in algs]...,
           Guide.manual_color_key("Method",collect(legend[algs]),collect(c.f(4)[algs])),
           Scale.y_log10,Stat.xticks(ticks=(collect(ranks))),
           Guide.xlabel("R"),Guide.ylabel("Time (s)"),Guide.Title("I=$S"));
    draw(SVG("$(dir)/time_rand$S.svg",12cm,8cm),p)
  end
end

@doc """ Function for testing algorithms HOSVD1, HOSVD2, HOSVD3, HOSVD4 on function-realted Tucker tensors of order 3.""" ->
function hosvdfunctest(;func=[1,2,3,4],algs=[1,2,3,4],create_plot=0,dir=".")
#Required:
#func ... subvector of [1,2,3,4] for predefined function-related tensors of size 50x50x50 (1), 100x100x100 (2),
#           200x200x200 (3), 400x400x400 (4); #e.g. func=[1,2] will only include tensors of size 50 and 100, func=[4] will
#            only include tensor of size 400.
#Optional:
#algs ... Algorithms HOSVD1, HOSVD2, HOSVD3, HOSVD4 defined by their numbers. If not set, all four algorithms will be tested,
#         e.g. hosvdtest("rand50",algs=[1,2]) will only test HOSVD1 and HOSVD2 algorithms.
#create_plot, dir ... Setting $create_plot to 1 will create plot from the results in the directory specified in $dir,
#                     e.g. hosvdtest("rand50", create_plot=1, dir="plots"). Make sure the directory exists!
#
#Example: hosvdfunctest()
#         hosvdfunctest(create_plot=1, dir="plots")
#         hosvdfunctest(func=[1,2], create_plot=1, dir="plots")

  println("\n\n**Initializing tensors.")

  L=length(func);
  nmbr=length(algs);
  S=[50,100,200,400];
  ranks=zeros(Int,L);
  println("\nCreating function related tensors.");

  T1=Array{Any}(L);T2=Array{Any}(L);
  i=1;
  for f in func
    T1[i]=load("test_data_func.jld","F$(2*f-1)");
    T2[i]=load("test_data_func.jld","F$(2*f)")
    println("-> Tucker tensors of size ",size(T1[i])," based on functions.")
    ranks[i]=coresize(T1[i])[1];
    i+=1;
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
      println("\nTwo function-related tensors of size $(S[func[i]]).");
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
      i+=1;
    end
  end

  if create_plot == 1
    println("\n\n**Creating plot.")

    c=Scale.color_discrete_hue() #manual colors

    timef1=zeros(nmbr);
    timef2=zeros(nmbr);
    timef3=zeros(nmbr);
    timef4=zeros(nmbr);
    i=1;
    for a in algs
      if 1 in func
        timef1[i]=(@eval $(Symbol("time_$a")))[1]
      end
      if 2 in func
        timef2[i]=(@eval $(Symbol("time_$a")))[2]
      end
      if 3 in func
        timef3[i]=@eval $(Symbol("time_$a"))[3]
      end
      if 4 in func
        timef4[i]=@eval $(Symbol("time_$a"))[4]
      end
      i+=1;
    end

    p1=plot(x=algs,y=timef1,color=algs,Geom.bar(position=:dodge),#Guide.xticks(label=false),
            Guide.xlabel(nothing),Guide.ylabel("Time (s)"),Guide.Title("I=$(S[1])"),
            Scale.x_discrete,Scale.color_discrete(c.f(4)), Theme(key_position = :none))
    p2=plot(x=algs,y=timef2,color=algs,Geom.bar(position=:dodge),#Guide.xticks(label=false),
            Guide.xlabel(nothing),Guide.ylabel(nothing),Guide.Title("I=$(S[2])"),
            Scale.x_discrete,Scale.color_discrete(c.f(4)), Theme(key_position = :none));
    p3=plot(x=algs,y=timef3,color=algs,Geom.bar(position=:dodge),#Guide.xticks(label=false),
            Guide.xlabel(nothing),Guide.ylabel(nothing),Guide.Title("I=$(S[3])"),
            Scale.x_discrete,Scale.color_discrete(c.f(4)), Theme(key_position = :none));
    p4=plot(x=algs,y=timef4,color=["HOSVD1","HOSVD2","HOSVD3","HOSVD4"],
            Guide.colorkey("Method"),Geom.bar(position=:dodge),#Guide.xticks(label=false),
            Guide.xlabel(nothing),Guide.ylabel(nothing),Guide.Title("I=$(S[4])"),
            Scale.x_discrete,Scale.color_discrete(c.f(4)));
    plt=hstack(compose(context(0, 0, 3.5cm, 8cm), render(p1)),
                 compose(context(0, 0, 2.7cm, 8cm), render(p2)),
                 compose(context(0, 0, 2.5cm, 8cm), render(p3)),
                 compose(context(0, 0, 3.9cm, 8cm), render(p4)));
    draw(SVG("$(dir)/time_func.svg",15cm,8cm),plt);
  end
end


@doc """ Function for testing algorithm HOSVD4 on Tucker tensors of order 3,
         with fixed multilinear rank (R,R,R) and different sizes SxSxS.""" ->
function hosvd4test{D<:Integer}(R,S::Vector{D};timeit_limit=500,create_plot=0,dir=".")
#Required:
#R ... multilinear rank (R,R,R)
#S ... vector of sizes [S1,S2,...,Sn]; creates tensors of multilinear ranks (Si,Si,Si), i=1,...,k
#Optional:
#timeit_limit ... Replaces @timeit function, which runs function in a loop, with @time, which runs function only once, if S>timeit_limit.
#create_plot, dir ... set $create_plot to 1 if you want to create plot that will be saved in the directory $dir
#                     (make sure the directory exists!)
#
#Example: hosvd4test(50,collect(500:500:5000),create_plot=1,dir="plots/")

  println("\n\n**Initializing tensors.")

  L=length(S);
  T1=Array{Any}(L);T2=Array{Any}(L);
  i=1;
  for sz in S
    println("\nCreating random tensors of size $S and ranks $R.");
    T1[i]=randttensor([sz,sz,sz],[R,R,R]);
    T2[i]=randttensor([sz,sz,sz],[R,R,R]);
    i+=1;
  end

  println("\n\n**Runing HOSVD4 algorithm.")

  times=zeros(L);
  i=1;
  for sz in S
    println("T=hosvd4(T1,T2,[$R,$R,$R]) ...")
    if sz >= timeit_limit && i>1
      tic(); T=hosvd4(T1[i],T2[i],reqrank=[R,R,R]); times[i]=toc();
	  else
      times[i]= @timeit T=hosvd4(T1[i],T2[i],reqrank=[R,R,R]);
    end
    i+=1
  end

  if create_plot == 1
    println("\n\n**Creating plot.")

    #manual colors:
    c=Scale.color_discrete_hue() #c.f(4);
    #for black&white:
    dash = 4 * Compose.mm
    ddot = .5 * Compose.mm
    gap = 1 * Compose.mm
    lines=[[],[dash],[dot],[dash,gap,ddot,gap]]
    legend=["\U2015\U2015\U2006 HOSVD1",
            "\U2013 \U2013 \U2013 HOSVD2",
            "\U2010\U2010\U2010\U2010\U2010\U2010 HOSVD3",
            "\U2006\U2013 \U2010 \U2013 HOSVD4"]

    p=plot(x=collect(S),y=times,Geom.line,Theme(default_color=color(c.f(4)[4]),line_style=[dash,gap,ddot,gap]),
           Guide.manual_color_key("Method",[legend[4]],[c.f(4)[4]]),
           Scale.y_log10,Stat.xticks(ticks=collect(S)),
           Guide.xlabel("I"),Guide.ylabel("Time (s)"),Guide.Title("R=$R"));
    draw(SVG("$(dir)/hosvd4_time_rand$R.svg",12cm,8cm),p)
  end

end
hosvd4test{D<:Integer}(R,S::Range{D};create_plot=0,dir=".")=hosvd4test(R,collect(S),create_plot=create_plot,dir=dir)
#

@doc """ Function for testing algorithms HOSVD1, HOSVD2, HOSVD3, HOSVD4 on randomly generated Tucker tensors of order N,
         size Sx...xS and different multilinear ranks (R1,...,R1) and (R2,...,R2), recompressing their Hadamard product
         to multilinear rank (R,...,R).""" ->
function hosvd_diffranks_test(S,ranks1,ranks2,ranksH,N;algs=[1,2,3,4],timeit_limit=500,hosvd2_limit=0,create_plot=0,dir=".")
#Required:
#S ... defines size of a tensors as Sx..xS
#ranks1 ... array of multilinear ranks [R1,R2,...,Rk] for first tensor; creates tensors of multilinear ranks (Ri,...,Ri), i=1,...,k
#ranks2 ... array of multilinear ranks [R1,R2,...,Rk] for second tensor; creates tensors of multilinear ranks (Ri,...,Ri), i=1,...,k
#ranksH ... array of multilinear ranks [R1,R2,...,Rk] for Hadamard product; creates tensors of multilinear ranks (Ri,...,Ri), i=1,...,k
#N ... order of tensors
#Optional:
#algs ... Algorithms HOSVD1, HOSVD2, HOSVD3, HOSVD4 defined by their numbers. If not set, all four algorithms will be tested,
#         e.g. hosvdtest(50,[2,4,6],algs=[1,2]) will only test HOSVD1 and HOSVD2 algorithms.
#timeit_limit ... Replaces @timeit function, which runs function in a loop, with @time, which runs function only one, if S>timeit_limit.
#create_plot, dir ... Setting $create_plot to 1 will create plot from the results in the directory specified in $dir,
#                     e.g. hosvdtest(50,[2,4,6], create_plot=1, dir="plots"). Make sure the directory exists!
#
#Example: hosvd_diffranks_test(50,collect(3:30),collect(9:3:90),3,collect(6:2:60),create_plot=1, dir="plots")

  @assert(length(ranks1)==length(ranks2)==length(ranksH),"Ranks need to be of same lenght!")
  L=length(ranks1);

  println("\n\n**Initializing tensors.")

  println("\nCreating random tensors of size $S and ranks $ranks1 and $ranks2.");
  T1=Array(Any,L);T2=Array(Any,L);
  for i=1:L
    T1[i]=randttensor(S,ranks1[i],N);
    T2[i]=randttensor(S,ranks2[i],N);
    println("-> Tucker tensors tensors of order $N, size $S and ranks $(ranks1[i]) and $(ranks2[i]).");
  end

  println("\n\n**Runing HOSVD algorithms.")

  for method in algs
    println("\nMethod hosvd$method.");

    time=Symbol("time_$(method)");
    err=Symbol("err_$(method)");

    @eval ($(time)=zeros($L));
    @eval ($(err)=zeros($L));

    for i=1:L
      R=ranksH[i];
      println("\nRecompression to rank $R.");
      if method==1
	      println("T=hosvd1(T1,T2,reqrank=$R) ...")
        if S >= timeit_limit && i>1
	        tic(); T=hosvd1(T1[i],T2[i],reqrank=R); @eval $(time)[i]=toc();
	      else
          @eval ($(time)[$i]= @timeit T=hosvd1($T1[$i],$T2[$i],reqrank=$R));
        end
      elseif method==2
        println("T=hosvd2(T1,T2,reqrank=$R) ...")
        if S >= timeit_limit && i>1
          tic(); T=hosvd2(T1[i],T2[i],reqrank=R); @eval $(time)[i]=toc();
	      else
          @eval ($(time)[$i]= @timeit T=hosvd2($T1[$i],$T2[$i],reqrank=$R));
        end
      elseif method == 3
        println("T=hosvd3(T1,T2,reqrank=$R) ...")
	      if R^2>S;var='A';else; var='B';end
	      if S >= timeit_limit && i>1
		      tic(); T=hosvd3(T1[i],T2[i],reqrank=R,variant=var); @eval $(time)[i]=toc();
	      else
          @eval ($(time)[$i]= @timeit T=hosvd3($T1[$i],$T2[$i],reqrank=$R,variant=$var));
        end
      elseif method==4
        println("T=hosvd4(T1,T2,reqrank=$R) ...")
        if S >= timeit_limit && i>1
          tic(); T=hosvd4(T1[i],T2[i],reqrank=R); @eval $(time)[i]=toc();
	      else
          @eval ($(time)[$i]= @timeit T=hosvd4($T1[$i],$T2[$i],reqrank=$R));
       end
      end
    end
  end

  if create_plot == 1
    println("\n\n**Creating plot.")
    #manual colors:
    c=Scale.color_discrete_hue() #c.f(4);
    #for black&white:
    dash = 4 * Compose.mm;
    ddot = .5 * Compose.mm;
    gap = 1 * Compose.mm;
    lines=[[],[dash],[ddot],[dash,gap,ddot,gap]]
    legend=["\U2015\U2015\U2006 HOSVD1",
            "\U2013 \U2013 \U2013 HOSVD2",
            "\U2010\U2010\U2010\U2010\U2010\U2010 HOSVD3",
            "\U2006\U2013 \U2010 \U2013 HOSVD4"]

    p=plot([layer(x=ranksH,y=(@eval $(Symbol("time_$a"))),Geom.line,
                  Theme(default_color=color(c.f(4)[a]),line_style=lines[a])) for a in algs]...,
                  Guide.manual_color_key("Method",collect(legend[algs]),collect(c.f(4)[algs])),
                   Scale.y_log10,Stat.xticks(ticks=(collect(ranksH))),
                   Guide.xlabel("R"),Guide.ylabel("Time (s)"),Guide.Title("I=$S"));

    draw(SVG("$dir/diff_ranks$S.svg",14cm,8cm),p)
  end
end

@doc """ Function for testing algorithms HOSVD1, HOSVD2, HOSVD3, HOSVD4 on randomly generated Tucker tensors of order 3,
         size S=S1xS2xS3 and different multilinear ranks P=(P1,P2,P3) and Q=(Q1,Q2,Q3), recompressing their Hadamard product
         to multilinear rank (R1,R2,R3).""" ->
function hosvd_diffmodes_test(S,P,Q,R;algs=[1,2,3,4],timeit_limit=500,hosvd2_limit=0,create_plot=0,dir=".")
#Required:
#S ... size of a tensors S=S1xS2xS3
#P ... multilinear rank of first tensor P=(P1,P2,P3)
#Q ... multilinear rank of second tensor Q=(Q1,Q2,Q3)
#R ... recompress HAdamard product to multilinear rank R=(R1,R2,R3)
#Optional:
#algs ... Algorithms HOSVD1, HOSVD2, HOSVD3, HOSVD4 defined by their numbers. If not set, all four algorithms will be tested,
#         e.g. hosvdtest(50,[2,4,6],algs=[1,2]) will only test HOSVD1 and HOSVD2 algorithms.
#timeit_limit ... Replaces @timeit function, which runs function in a loop, with @time, which runs function only one, if S>timeit_limit.
#create_plot, dir ... Setting $create_plot to 1 will create plot from the results in the directory specified in $dir,
#                     e.g. hosvdtest(50,[2,4,6], create_plot=1, dir="plots"). Make sure the directory exists!
#
#Example: hosvd_diffmodes_test([200,200,200],[2,20,40],[40,20,2],[20,20,20],create_plot=1, dir="plots")

  println("\n\n**Initializing tensors.")
  println("\nCreating random tensors...");

  nmbr=length(algs);

  T1=randttensor(S,P);
  T2=randttensor(S,Q);
  println("\nTwo random tensors of order size $S and ranks $P and $Q.");

  println("\n\n**Runing HOSVD algorithms.")

   for method in algs

    println("\nMethod hosvd$method");

    time=Symbol("time_$(method)");
    err=Symbol("err_$(method)");

    println("\nTwo random tensors of size $S and ranks $P and $Q.");

    if method==1
	    println("T=hosvd1(T1,T2,$R) ...")
      if maximum(S) >=timeit_limit && i>1
        tic(); T=hosvd1(T1,T2,reqrank=R); @eval $time=toc();
	    else
        @eval ($time= @timeit T=hosvd1($T1,$T2,reqrank=$R));
      end
    elseif method==2
      println("T=hosvd2(T1,T2,$R) ...")
      if maximum(S)>timeit_limit && i>1
        tic(); T=hosvd2(T1,T2,reqrank=R); @eval $time=toc();
	    else
        @eval ($time= @timeit T=hosvd2($T1,$T2,reqrank=$R));
      end
    elseif method == 3
      println("T=hosvd3(T1,T2,$R) ...")
	    if maximum(P.*Q)>maximum(S);var='A';else; var='B';end
	    if maximum(S)>timeit_limit && i>1
		    tic(); T=hosvd3(T1,T2,reqrank=R,variant=var); @eval $time=toc();
	    else
        @eval ($time= @timeit T=hosvd3($T1,$T2,reqrank=$R,variant=$var));
      end
    elseif method==4
      println("T=hosvd4(T1,T2,$R) ...")
      if maximum(S)>timeit_limit && i>1
        tic(); T=hosvd4(T1,T2,reqrank=R); @eval $time=toc();
	    else
        @eval ($time= @timeit T=hosvd4($T1,$T2,reqrank=$R));
      end
    end
  end
  if create_plot == 1
    println("\n\n**Creating plot.")

    #manual colors:
    c=Scale.color_discrete_hue() #c.f(4);
    #for black&white:
    dash = 4 * Compose.mm
    ddot = .5 * Compose.mm
    gap = 1 * Compose.mm
    lines=[[],[dash],[ddot],[dash,gap,ddot,gap]]
    legend=["\U2015\U2015\U2006 HOSVD1",
            "\U2013 \U2013 \U2013 HOSVD2",
            "\U2010\U2010\U2010\U2010\U2010\U2010 HOSVD3",
            "\U2006\U2013 \U2010 \U2013 HOSVD4"]

    time=zeros(nmbr);
    i=1;
    for a in algs
      time[i]=(@eval $(Symbol("time_$a")))[1]
      i+=1;
    end

    p=plot(x=algs,y=time,color=["HOSVD1","HOSVD2","HOSVD3","HOSVD4"],Guide.colorkey("Method"),
	          Geom.bar(position=:dodge),#Guide.xticks(label=false),
            Guide.xlabel(nothing),Guide.ylabel("Time (s)"),Guide.Title("$(S[1])x$(S[2])x$(S[3])"),
            Scale.x_discrete,Scale.color_discrete(c.f(4)))
    draw(SVG("$dir/diff_modes$S.svg",8cm,8cm),p)
  end
end