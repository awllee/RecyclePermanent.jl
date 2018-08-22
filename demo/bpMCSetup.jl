using RecyclePermanent
using RNGPool
using JLD2
using StaticArrays
using LinearAlgebra
using Random
import MonteCarloMarkovKernels: simulateChainProgress, makeAMKernel

include("../models/bpModel.jl")

@load "data/bpData.jld" ys θ
θ0 = θ

d = 3

@inline function toBPθ(v::SVector{d, Float64})
  return BPθ(v[1], v[2], v[3])
end

t0 = SVector{d, Float64}(θ0.λ, θ0.a, θ0.b)

sigmaProp = 1.0*SMatrix{d, d, Float64}(Matrix{Float64}(I, d, d))

@inline function lglogprior(θ::BPθ)
  if θ.λ < 0 return -Inf end
  if θ.a < 0 return -Inf end
  if θ.b < 0 return -Inf end
  v::Float64 = 0.0
  v -= 0.001*θ.λ
  v -= 0.1*θ.a
  v -= 0.1*θ.b
  return v
end

N = 40 * length(ys)
chainLength = 1000000
