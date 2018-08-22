@inline function simpleEstimate(model::Model, rpio::RPIO{Particle}) where Particle
  potentials = rpio.potentials
  v::Float64 = 0.0
  for p in 1:rpio.n
    _sampleParticles(model, rpio)
    v += log(_evaluatePotentials(model, rpio, p) / rpio.N)
  end
  return v
end
