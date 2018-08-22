using RecyclePermanent
using RNGPool
using JLD2

include("../models/gkModel.jl")

setRNGs(12345)

θ = GKθ(3.0, 1.0, 2.0, 0.5)
model = makeGKModel(θ)
ys = simulateModel(model, 1000)

@save "data/gkData.jld" ys θ
