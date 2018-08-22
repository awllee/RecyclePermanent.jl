using RecyclePermanent
using RNGPool
using JLD2
using DataFrames
using BenchmarkTools
include("../models/gkModel.jl")
include("compare.jl")

@load "data/gkData.jld" ys θ
model = makeGKModel(θ, ys)

@load "compare/gktau.jld" ns trials outR outS outKM1 outKM2 outKM3

PR = Int.(round.((x->x[1]).(outR), sigdigits=2))
NS = Int.(round.((x->x[1]).(outS), sigdigits=2))
PS = Int.(round.(ns .*(x->x[1]).(outS), sigdigits=2))
PKM1 = Int.(round.((x->x[1]).(outKM1), sigdigits=2))
PKM2 = Int.(round.((x->x[1]).(outKM2), sigdigits=2))
PKM3 = Int.(round.((x->x[1]).(outKM3), sigdigits=2))

df = DataFrame(n = ns, PR = PR, PS = PS, PKM1 = PKM1, PKM2 = PKM2, PKM3 = PKM3)
show(df)
println()

m = length(ns)

tR = (i -> @belapsed rpEstimate(model, $(RPIO{model.Tx}(PR[i],ns[i],1)))).(1:m)
tS = (i -> @belapsed simpleEstimate(model, $(RPIO{model.Tx}(NS[i],ns[i],1)))).(1:m)
tKM1 = (i -> @belapsed kmEstimate(model, $(KMIO{model.Tx}(PKM1[i],ns[i])),1.0)).(1:m)
tKM2 = (i -> @belapsed kmEstimate(model, $(KMIO{model.Tx}(PKM2[i],ns[i])),0.001)).(1:m)
tKM3 = (i -> @belapsed kmEstimate(model, $(KMIO{model.Tx}(PKM3[i],ns[i])),0.0001)).(1:m)

TR = round.(tR*1000, digits=1)
TS = round.(tS*1000, digits=1)
TKM1 = round.(tKM1*1000, digits=1)
TKM2 = round.(tKM2*1000, digits=1)
TKM3 = round.(tKM3*1000, digits=1)

df2 = DataFrame(n = ns, TR = TR, TS = TS, TKM1 = TKM1, TKM2 = TKM2, TKM3 = TKM3)
show(df2)
println()
