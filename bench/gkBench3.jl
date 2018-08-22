using RecyclePermanent
using BenchmarkTools
using JLD2
include("../models/gkModel.jl")

@load "data/gkData.jld" ys θ
ys = ys[1:100]
model = makeGKModel(θ, ys)

N = 80 * model.n

rpio = RPIO{model.Tx}(N, model.n, Threads.nthreads())
@btime simpleEstimate($model, $rpio)
@btime biasedEstimate($model, $rpio)
@btime rpEstimate($model, $rpio)

rpio = RPIO{model.Tx}(N, model.n, 1)
@btime simpleEstimate($model, $rpio)
@btime biasedEstimate($model, $rpio)
@btime rpEstimate($model, $rpio)
