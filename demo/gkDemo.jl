using RecyclePermanent
using JLD2

include("../models/gkModel.jl")

@load "data/gkData.jld" ys θ
ys = ys[1:100]
model = makeGKModel(θ, ys)

N = 80 * model.n

println("Single-threaded")
rpio = RPIO{model.Tx}(N, model.n, 1)
@time simpleEstimate(model, rpio)
@time simpleEstimate(model, rpio)
@time biasedEstimate(model, rpio)
@time biasedEstimate(model, rpio)
@time rpEstimate(model, rpio)
@time rpEstimate(model, rpio)

println("Multi-threaded")
rpio = RPIO{model.Tx}(N, model.n, Threads.nthreads())
@time simpleEstimate(model, rpio)
@time simpleEstimate(model, rpio)
@time biasedEstimate(model, rpio)
@time biasedEstimate(model, rpio)
@time rpEstimate(model, rpio)
@time rpEstimate(model, rpio)
