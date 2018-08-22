using RecyclePermanent
using MonteCarloMarkovKernels: simulateChainProgress, makeAMKernel, estimateBM,
  kde
using StaticArrays
using StatsBase
using LinearAlgebra
using JLD2
using Plots
Plots.gr()

include("../models/gaussianModel.jl")

@load "data/gaussianData.jld" ys θ
ys = ys[1:10]

d = 2

@inline function toGaussianθ(v::SVector{d, Float64})
  return Gaussianθ(v[1], v[2], θ.C, θ.D)
end

θ0 = SVector{d, Float64}(θ.A, θ.B)

const sigmaProp = SMatrix{d, d, Float64}(Matrix{Float64}(I, d, d))

@inline function lglogprior(theta::Gaussianθ)
  if theta.A < 0 || theta.A > 5 return -Inf end
  if theta.B < 0 || theta.B > 5 return -Inf end
  return 0.0
end

function makeSimpleLTD(ys::Vector{Float64}, N::Int64)
  rpio::RPIO{Float64Particle} = RPIO{Float64Particle}(N, length(ys), 1)
  function ltd(in::SVector{d, Float64})
    theta::Gaussianθ = toGaussianθ(in)
    lp::Float64 = lglogprior(theta)
    if lp == -Inf return -Inf end
    model::Model = makeGaussianModel(theta, ys)
    return simpleEstimate(model, rpio)
  end
end

function makeRecycleLTD(ys::Vector{Float64}, N::Int64)
  rpio::RPIO{Float64Particle} = RPIO{Float64Particle}(N, length(ys), 1)
  function ltd(in::SVector{d, Float64})
    theta::Gaussianθ = toGaussianθ(in)
    lp::Float64 = lglogprior(theta)
    if lp == -Inf return -Inf end
    model::Model = makeGaussianModel(theta, ys)
    return rpEstimate(model, rpio)
  end
end

function makeExactLTD(ys::Vector{Float64})
  function ltd(in::SVector{d, Float64})
    theta::Gaussianθ = toGaussianθ(in)
    lp::Float64 = lglogprior(theta)
    if lp == -Inf return -Inf end
    return gaussianExactLL(theta, ys)
  end
end

N = 40 * length(ys)
chainLength = 1000000

logtargetSimple = makeSimpleLTD(ys, N)
logtargetRecycle = makeRecycleLTD(ys, N)
logtargetExact = makeExactLTD(ys)

PSimple = makeAMKernel(logtargetSimple, sigmaProp)
PRecycle = makeAMKernel(logtargetRecycle, sigmaProp)
PExact = makeAMKernel(logtargetExact, sigmaProp)

Random.seed!(12345)

@time chainSimple = simulateChainProgress(PSimple, θ0, chainLength)
sar = PSimple(:acceptanceRate)

@time chainRecycle = simulateChainProgress(PRecycle, θ0, chainLength)
rar = PRecycle(:acceptanceRate)

@time chainExact = simulateChainProgress(PExact, θ0, chainLength)
ear = PExact(:acceptanceRate)

vsSimple = (i->(x->x[i]).(chainSimple)).(1:d)
vsRecycle = (i->(x->x[i]).(chainRecycle)).(1:d)
vsExact = (i->(x->x[i]).(chainExact)).(1:d)

plot(kde(vsSimple[1], sar))
plot!(kde(vsExact[1], ear))
plot!(kde(vsRecycle[1], rar))

plot(kde(vsSimple[2], sar))
plot!(kde(vsExact[2], ear))
plot!(kde(vsRecycle[2], rar))

contour(kde(vsSimple[1], vsSimple[2], sar))
contour!(kde(vsExact[1], vsExact[2], ear))
contour!(kde(vsRecycle[1], vsRecycle[2], rar))

plot(autocor(vsSimple[1]))
plot!(autocor(vsExact[1]))
plot!(autocor(vsRecycle[1]))

plot(autocor(vsSimple[2]))
plot!(autocor(vsExact[2]))
plot!(autocor(vsRecycle[2]))

avarsSimple = estimateBM.(vsSimple)
avarsRecycle = estimateBM.(vsRecycle)
avarsExact = estimateBM.(vsExact)
