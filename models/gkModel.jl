using Random

struct GKθ
   A::Float64
   B::Float64
   g::Float64
   k::Float64
end

const GKϵ = 0.2
const GKc = 0.8

mutable struct Float64Particle
  x::Float64
  Float64Particle() = new()
end

function makeGKModel(θ::GKθ, ys::Vector{Float64} = Vector{Float64}(undef, 0))
  n::Int64 = length(ys)
  @inline function G(p::Int64, px::Float64Particle)
    x::Float64 = px.x
    @inbounds return abs(x - ys[p]) < GKϵ
  end
  @inline function μ!(p::Float64Particle, rng::RNG) where RNG <: AbstractRNG
    z::Float64 = randn(rng)
    t1::Float64 = exp(-θ.g*z)
    p.x = θ.A + θ.B*(1+GKc*(1-t1)/(1+t1))*(1+z*z)^θ.k*z
  end
  @inline function simulate(p::Float64Particle, rng::RNG) where RNG<:AbstractRNG
    x::Float64 = p.x
    return x + (rand(rng)*2-1)*GKϵ
  end
  return Model(μ!, G, n, simulate, Float64Particle, Float64)
end
