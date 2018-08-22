using RecyclePermanent
using RNGPool
using JLD2

include("../models/gaussianModel.jl")

setRNGs(12345)

θ = Gaussianθ(0.0, 1.5, 2.5, 0.5)
model = makeGaussianModel(θ)
ys = simulateModel(model, 1000)

@save "data/gaussianData.jld" ys θ
