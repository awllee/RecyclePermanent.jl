using RecyclePermanent
using JLD2

include("../models/bpModel.jl")

@load "data/bpData.jld" ys θ
model = makeBPModel(θ, ys)

N = 40 * model.n

println("Single-threaded")
rpio = RPIO{model.Tx}(N, model.n, 1)
@time simpleEstimate(model, rpio)
@time simpleEstimate(model, rpio)
@time biasedEstimate(model, rpio)
@time biasedEstimate(model, rpio)
@time biasedEstimateDiscrete(model, rpio)
@time biasedEstimateDiscrete(model, rpio)
@time rpEstimate(model, rpio)
@time rpEstimate(model, rpio)
@time rpEstimateDiscrete(model, rpio)
@time rpEstimateDiscrete(model, rpio)

println("Multi-threaded")
rpio = RPIO{model.Tx}(N, model.n, Threads.nthreads())
@time simpleEstimate(model, rpio)
@time simpleEstimate(model, rpio)
@time biasedEstimate(model, rpio)
@time biasedEstimate(model, rpio)
@time biasedEstimateDiscrete(model, rpio)
@time biasedEstimateDiscrete(model, rpio)
@time rpEstimate(model, rpio)
@time rpEstimate(model, rpio)
@time rpEstimateDiscrete(model, rpio)
@time rpEstimateDiscrete(model, rpio)
