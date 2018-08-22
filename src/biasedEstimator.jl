@inline function biasedEstimate(model::Model, rpio::RPIO{Particle}) where Particle
  potentials::Vector{Float64} = rpio.potentials
  v::Float64 = 0.0
  _sampleParticles(model, rpio)
  for p in 1:rpio.n
    v += log(_evaluatePotentials(model, rpio, p) / rpio.N)
  end
  return v
end

@inline function biasedEstimateDiscrete(model::Model, rpio::RPIO{Int64Particle})
  _sampleParticles(model, rpio)
  ζcounts::Vector{Int64} = _getCounts(rpio)
  potentials::Vector{Float64} = Vector{Float64}(undef, length(ζcounts))
  v::Float64 = 0.0
  for p in 1:rpio.n
    s::Float64 = _countPotentials(potentials, ζcounts, model, rpio, p)
    v += log(s / rpio.N)
  end
  return v
end
