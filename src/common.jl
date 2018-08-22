@inline function _sampleParticlesSerial(rpio::RPIO{Particle}, μ!::F) where
  {Particle, F<:Function}
  particles::Vector{Particle} = rpio.particles
  N::Int64 = rpio.N
  rng::RNG = getRNG()
  for i in 1:N
    @inbounds μ!(particles[i], rng)
  end
end

@inline function _sampleParticlesParallel(rpio::RPIO{Particle}, μ!::F) where {Particle, F<:Function}
  particles::Vector{Particle} = rpio.particles
  N::Int64 = rpio.N
  nt::Int64 = rpio.threads
  b::Int64 = div(N, nt)
  Threads.@threads for i in 1:nt
    start::Int64 = (i-1)*b + 1
    finish::Int64 = i != nt ? i*b : N
    rng::RNG = getRNG()
    for j in start:finish
      @inbounds μ!(particles[j], rng)
    end
  end
end

@inline function _sampleParticles(model::Model, rpio::RPIO{Particle}) where Particle
  if rpio.threads > 1
    _sampleParticlesParallel(rpio, model.μ!)
  else
    _sampleParticlesSerial(rpio, model.μ!)
  end
end

@inline function _evaluatePotentialsSerial(rpio::RPIO{Particle}, p::Int64,
   G::F) where {Particle, F<:Function}
  particles::Vector{Particle} = rpio.particles
  potentials::Vector{Float64} = rpio.potentials
  v::Float64 = 0.0
  for i in 1:length(potentials)
    @inbounds potentials[i] = G(p, particles[i])
    @inbounds v += potentials[i]
  end
  return v
end

@inline function _evaluatePotentialsParallel(rpio::RPIO{Particle}, p::Int64,
   G::F) where {Particle, F<:Function}
  particles::Vector{Particle} = rpio.particles
  potentials::Vector{Float64} = rpio.potentials
  scratch::Vector{Float64} = rpio.scratch
  N::Int64 = rpio.N
  nt::Int64 = rpio.threads
  b::Int64 = div(N, nt)
  Threads.@threads for i in 1:nt
    start::Int64 = (i-1)*b + 1
    finish::Int64 = i != nt ? i*b : N
    v::Float64 = 0.0
    for j in start:finish
      @inbounds potentials[j] = G(p, particles[j])
      @inbounds v += potentials[j]
    end
    @inbounds scratch[i] = v
  end
  return sum(scratch)
end

 @inline function _evaluatePotentials(model::Model, rpio::RPIO{Particle},
   p::Int64) where Particle
   if rpio.threads > 1
     return _evaluatePotentialsParallel(rpio, p, model.G)
   else
     return _evaluatePotentialsSerial(rpio, p, model.G)
   end
 end

@inline function sampleOne(xs::Vector{Float64}, sumXs::Float64,
  start::Int64 = 1)
  rng = getRNG()
  u::Float64 = rand(rng)*sumXs
  i::Int64 = start
  v::Float64 = xs[i]
  while u > v
    i += 1
    @inbounds v += xs[i]
  end
  return i
end
