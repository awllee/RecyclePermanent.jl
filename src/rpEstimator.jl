import NonUniformRandomVariateGeneration.sampleCategorical

@inline function sampleOne(xs::Vector{Float64}, sumXs::Float64, rpio::RPIO)
  M::Int64 = length(xs)
  if rpio.threads > 1 && M > PARALLEL_COUNTS_THRESHOLD
    # rpio.scratch contains the summed weights for each block
    idx::Int64 = sampleOne(rpio.scratch, sumXs)
    nt::Int64 = rpio.threads
    b::Int64 = div(M, nt)
    start::Int64 = (idx-1)*b + 1
    @inbounds return sampleOne(xs, rpio.scratch[idx], start)
  else
    return sampleCategorical(xs, sumXs)
  end
end

@inline function rpEstimate(model::Model, rpio::RPIO{Particle}) where Particle
  potentials::Vector{Float64} = rpio.potentials
  ks::Vector{Int64} = rpio.ks
  v::Float64 = 0.0
  _sampleParticles(model, rpio)
  b::Int64 = div(rpio.N, rpio.threads)
  for p in 1:rpio.n
    s::Float64 = _evaluatePotentials(model, rpio, p)
    for j in 1:p-1
      @inbounds block::Int64 = min(ceil(ks[j]/b), rpio.threads)
      @inbounds rpio.scratch[block] -= potentials[ks[j]]
      @inbounds s -= potentials[ks[j]]
      @inbounds potentials[ks[j]] = 0.0
    end
    v += log(s / (rpio.N-p+1))
    if p < model.n
      @inbounds ks[p] = sampleOne(potentials, s, rpio)
    end
  end
  return v
end

@inline function rpEstimateDiscrete(model::Model, rpio::RPIO{Int64Particle})
  _sampleParticles(model, rpio)
  ζcounts::Vector{Int64} = _getCounts(rpio)
  potentials::Vector{Float64} = Vector{Float64}(undef, length(ζcounts))

  v::Float64 = 0.0
  for p in 1:rpio.n
    s::Float64 = _countPotentials(potentials, ζcounts, model, rpio, p)
    v += log(s / (rpio.N-p+1))
    if p < model.n
      k::Int64 = sampleOne(potentials, s, rpio)
      @inbounds ζcounts[k] -= 1
    end
  end
  return v
end
