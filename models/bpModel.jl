using Random
import NonUniformRandomVariateGeneration: sampleBeta, samplePoisson

struct BPθ
   λ::Float64
   a::Float64
   b::Float64
end

const BPσ = 5

function makeBPModel(θ::BPθ, ys::Vector{Float64} = Vector{Float64}(undef, 0))
  n::Int64 = length(ys)
  invσ²over2::Float64 = 0.5/BPσ^2
  logncG::Float64 = -0.5 * log(2 * π * BPσ^2)
  @inline function G(p::Int64, px::Int64Particle)
    @inbounds v::Float64 = px.x - ys[p]
    lG::Float64 = logncG - v * invσ²over2 * v
    @inbounds return exp(lG)
  end
  @inline function μ!(p::Int64Particle, rng::RNG) where RNG <: AbstractRNG
    S::Float64 = sampleBeta(θ.a, θ.b, rng)
    p.x = samplePoisson(θ.λ*S, rng)
  end
  @inline function simulate(p::Int64Particle, rng::RNG) where RNG <: AbstractRNG
    return p.x + BPσ*randn(rng)
  end
  return Model(μ!, G, n, simulate, Int64Particle, Float64)
end
