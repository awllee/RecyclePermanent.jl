using RecyclePermanent
using JLD2
using BenchmarkTools

include("../models/gkModel.jl")
include("bench.jl")

@load "data/gkData.jld" ys θ
ys = ys[1:100]
model = makeGKModel(θ, ys)

N = 80*model.n

trials = 1000

rpio = RPIO{model.Tx}(N, model.n, Threads.nthreads())
testEstimator(() -> simpleEstimate(model, rpio), trials)
testEstimator(() -> biasedEstimate(model, rpio), trials)
testEstimator(() -> rpEstimate(model, rpio), trials)

rpio = RPIO{model.Tx}(N, model.n, 1)
testEstimator(() -> simpleEstimate(model, rpio), trials)
testEstimator(() -> biasedEstimate(model, rpio), trials)
testEstimator(() -> rpEstimate(model, rpio), trials)
