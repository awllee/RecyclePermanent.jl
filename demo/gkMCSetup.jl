using RecyclePermanent
using RNGPool
using JLD2
using StaticArrays
using Random
using LinearAlgebra
import MonteCarloMarkovKernels: simulateChainProgress, makeAMKernel

include("../models/gkModel.jl")

@load "data/gkData.jld" ys θ
ys = ys[1:100]
θ0 = θ

d = 4

@inline function toGKθ(v::SVector{d, Float64})
  return GKθ(v[1], v[2], v[3], v[4])
end

t0 = SVector{d, Float64}(θ0.A, θ0.B, θ0.g, θ0.k)

sigmaProp = 1.0*SMatrix{d, d, Float64}(Matrix{Float64}(I, d, d))

@inline function lglogprior(θ::GKθ)
  if θ.A < 0 || θ.A > 10 return -Inf end
  if θ.B < 0 || θ.B > 10 return -Inf end
  if θ.g < 0 || θ.g > 10 return -Inf end
  if θ.k < 0 || θ.k > 10 return -Inf end
  return 0.0
end

N = 80 * length(ys)
chainLength = 1000000
