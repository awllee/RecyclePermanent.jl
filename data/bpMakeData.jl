using RecyclePermanent
using RNGPool
using JLD2

include("../models/bpModel.jl")

setRNGs(12345)

θ = BPθ(500, 2, 8)
model = makeBPModel(θ)
ys = simulateModel(model, 1000)

@save "data/bpData.jld" ys θ
