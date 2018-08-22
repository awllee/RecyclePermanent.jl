using RecyclePermanent
using RNGPool
using JLD2
using BenchmarkTools

include("../models/bpModel.jl")

@load "data/bpData.jld" ys θ
model = makeBPModel(θ, ys)

N = 1024*1024
xs = Vector{Int64Particle}(undef, N)
for i in 1:N
  xs[i] = Int64Particle()
end
gs = Vector{Float64}(undef, N)

function fillμ(μ!::F, xs::Vector{Int64Particle}) where F<:Function
  rng = getRNG()
  for i in 1:length(xs)
    @inbounds μ!(xs[i], rng)
  end
end

function fillG(G::F, xs::Vector{Int64Particle}, gs::Vector{Float64}) where
  F<:Function
  for i in 1:length(xs)
    @inbounds gs[i] = G(1, xs[i])
  end
end

@btime fillμ($model.μ!, $xs)
@btime fillG($model.G, $xs, $gs)
