using RecyclePermanent
using RNGPool
using JLD2
using BenchmarkTools

include("../models/bpModel.jl")
include("compare.jl")

@load "data/bpData.jld" ys θ
model = makeBPModel(θ, ys)

N = 40*model.n

trials = 1000

setRNGs(12345)

n = 10
N = 40*n

rpio = RPIO{model.Tx}(N, n, 1)

# simpleOutput = testEstimator(() -> simpleEstimate(model, rpio), trials)
biasedOutput = testEstimator(() -> biasedEstimateDiscrete(model, rpio), trials)
rpOutput = testEstimator(() -> rpEstimateDiscrete(model, rpio), trials)

ϵ = 1e-5
kmio = KMIO{model.Tx}(N, n)
kmOutput = testEstimator(() -> kmEstimate(model, kmio, ϵ), trials)

@btime rpEstimateDiscrete($model, $rpio)
