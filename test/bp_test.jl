using JLD2
include("../models/bpModel.jl")

setRNGs(12345)

@load "data/bpData.jld" ys θ
model = makeBPModel(θ, ys)

N = 400 * model.n

rpio = RPIO{model.Tx}(N, model.n, 1)
@test rpEstimateDiscrete(model, rpio) ≈ -5415.0 atol=1.0
rpio = RPIO{model.Tx}(N, model.n, 4)
@test rpEstimateDiscrete(model, rpio) ≈ -5415.0 atol=1.0
