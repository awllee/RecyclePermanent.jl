using RecyclePermanent
using RNGPool
using JLD2
using BenchmarkTools

include("../models/gkModel.jl")

@load "data/gkData.jld" ys θ
ys = ys[1:100]
model = makeGKModel(θ, ys)

N = 80 * model.n
xs = Vector{Float64Particle}(undef, N)
for i in 1:N
  xs[i] = Float64Particle()
end
gs = Vector{Float64}(undef, N)

function fillμ(μ!::F, xs::Vector{Float64Particle}) where F<:Function
  rng= getRNG()
  for i in 1:length(xs)
    @inbounds μ!(xs[i], rng)
  end
end

function fillG(G::F, xs::Vector{Float64Particle}, gs::Vector{Float64}) where
  F<:Function
  for i in 1:length(xs)
    @inbounds gs[i] = G(1, xs[i])
  end
end

@btime fillμ($model.μ!, $xs)
@btime fillG($model.G, $xs, $gs)
