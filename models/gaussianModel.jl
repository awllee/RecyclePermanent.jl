using Random

struct Gaussianθ
   A::Float64
   B::Float64
   C::Float64
   D::Float64
end

mutable struct Float64Particle
  x::Float64
  Float64Particle() = new()
end

function makeGaussianModel(θ::Gaussianθ, ys::Vector{Float64} = Vector{Float64}(undef, 0))
  n::Int64 = length(ys)
  sqrtB::Float64 = sqrt(θ.B)
  invDover2::Float64 = 0.5/θ.D
  logncG::Float64 = -0.5 * log(2 * π * θ.D)
  @inline function G(p::Int64, px::Float64Particle)
    @inbounds v::Float64 = θ.C * px.x - ys[p]
    lG::Float64 = logncG - v * invDover2 * v
    return exp(lG)
  end
  @inline function μ!(px::Float64Particle, rng::RNG) where RNG <: AbstractRNG
    px.x = θ.A + sqrtB*randn(rng)
  end
  @inline function simulate(px::Float64Particle, rng::RNG) where RNG <: AbstractRNG
    return θ.C*px.x + sqrt(θ.D)*randn(rng)
  end
  return Model(μ!, G, n, simulate, Float64Particle, Float64)
end

@inline function ldnorm(x, mu, var)
  lnc = -0.5*log(2*π*var)
  return lnc - 0.5/var*(x-mu)^2
end

function gaussianExactLL(θ::Gaussianθ, ys::Vector{Float64})
  n::Int64 = length(ys)
  v::Float64 = 0.0
  for p in 1:n
    v += ldnorm(ys[p], θ.A*θ.C, θ.C^2*θ.B+θ.D)
  end
  return v
end

# function defaultGaussianModel(n::Int64)
#   θ = Gaussianθ(0.0, 1.5, 2.5, 0.5)
#   model = makeGaussianModel(θ)
#   ys = simulateModel(model, n)
#   model = makeGaussianModel(θ, ys)
#   exactLL = gaussianExactLL(θ, ys)
#   return model, θ, ys, exactLL
# end
