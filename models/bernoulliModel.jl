using Random

function makeBernoulliModel(θ::Float64, ys::Vector{Int64} = Vector{Int64}(undef, 0))
  n::Int64 = length(ys)
  @inline function G(p::Int64, px::Int64Particle)
    @inbounds return px.x == ys[p] ? 1.0 : 0.0
  end
  @inline function μ(px::Int64Particle, rng::RNG) where RNG <: AbstractRNG
    px.x = rand(rng) < θ ? 1 : 0
  end
  @inline function simulate(px::Int64Particle, ::RNG) where RNG <: AbstractRNG
    return px.x
  end
  return Model(μ, G, n, simulate, Int64Particle, Int64)
end
