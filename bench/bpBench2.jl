using RecyclePermanent
using JLD2
using BenchmarkTools

include("../models/bpModel.jl")
include("bench.jl")

@load "data/bpData.jld" ys θ
model = makeBPModel(θ, ys)

N = 40*model.n

trials = 1000

rpio = RPIO{model.Tx}(N, model.n, 1)

testEstimator(() -> simpleEstimate(model, rpio), trials)
testEstimator(() -> biasedEstimate(model, rpio), trials)
testEstimator(() -> rpEstimate(model, rpio), trials)
testEstimator(() -> biasedEstimateDiscrete(model, rpio), trials)
testEstimator(() -> rpEstimateDiscrete(model, rpio), trials)
