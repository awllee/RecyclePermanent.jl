using RecyclePermanent
using BenchmarkTools
using JLD2
include("../models/bpModel.jl")

@load "data/bpData.jld" ys θ
model = makeBPModel(θ, ys)

N = 40 * model.n

rpio = RPIO{model.Tx}(N, model.n, Threads.nthreads())
@btime simpleEstimate($model, $rpio)
@btime biasedEstimate($model, $rpio)
@btime biasedEstimateDiscrete($model, $rpio)
@btime rpEstimate($model, $rpio)
@btime rpEstimateDiscrete($model, $rpio)

rpio = RPIO{model.Tx}(N, model.n, 1)
@btime simpleEstimate($model, $rpio)
@btime biasedEstimate($model, $rpio)
@btime biasedEstimateDiscrete($model, $rpio)
@btime rpEstimate($model, $rpio)
@btime rpEstimateDiscrete($model, $rpio)
