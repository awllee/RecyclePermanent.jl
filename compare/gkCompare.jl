using RecyclePermanent
using RNGPool
using JLD2
using BenchmarkTools

include("../models/gkModel.jl")
include("compare.jl")

@load "data/gkData.jld" ys θ
ys = ys[1:100]
model = makeGKModel(θ, ys)

N = 80*model.n

trials = 1000

setRNGs(12345)

n = 100
N = 50*n

rpio = RPIO{model.Tx}(N, n, 1)

simpleOutput = testEstimator(() -> simpleEstimate(model, rpio), trials)
biasedOutput = testEstimator(() -> biasedEstimate(model, rpio), trials)
rpOutput = testEstimator(() -> rpEstimate(model, rpio), trials)

kmio = KMIO{model.Tx}(N, n)
# ϵ = 1e-2
ϵ = 1e-5
kmEstimate(model, kmio, ϵ)
kmOutput = testEstimator(() -> kmEstimate(model, kmio, ϵ), trials)

@btime rpEstimate($model, $rpio)
