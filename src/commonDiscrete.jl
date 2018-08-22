@inline function _countPotentialsSerial(potentials::Vector{Float64},
  ζcounts::Vector{Int64}, rpio::RPIO{Int64Particle}, p::Int64, G::F) where F<:Function
  v::Float64 = 0.0
  px::Int64Particle = rpio.scratchInt64Particle[1]
  for i in 1:length(ζcounts)
    @inbounds if ζcounts[i] > 0
      px.x = i-1
      @inbounds potentials[i] = ζcounts[i] * G(p, px)
      # @inbounds potentials[i] = ζcounts[i] * G(p, i-1)
      @inbounds v += potentials[i]
    else
      @inbounds potentials[i] = 0.0
    end
  end
  return v
end

@inline function _countPotentialsParallel(potentials::Vector{Float64},
  ζcounts::Vector{Int64}, rpio::RPIO{Int64Particle}, p::Int64, G::F) where F<:Function
  scratch::Vector{Float64} = rpio.scratch
  M::Int64 = length(ζcounts)
  nt::Int64 = rpio.threads
  b::Int64 = div(M, nt)
  Threads.@threads for i in 1:nt
    start::Int64 = (i-1)*b + 1
    finish::Int64 = i != nt ? i*b : M
    v::Float64 = 0.0
    px::Int64Particle = rpio.scratchInt64Particle[i]
    for j in start:finish
      @inbounds if ζcounts[j] > 0
        px.x = j-1
        @inbounds potentials[j] = ζcounts[j] * G(p, px)
        # @inbounds potentials[j] = ζcounts[j] * G(p, j-1)
        @inbounds v += potentials[j]
      else
        @inbounds potentials[j] = 0.0
      end
    end
    @inbounds scratch[i] = v
  end
  return sum(scratch)
end

const PARALLEL_COUNTS_THRESHOLD = 256

function maximumI64(ps::Vector{Int64Particle})
  m::Int64 = 0
  for i in eachindex(ps)
    @inbounds m = max(m, ps[i].x)
  end
  return m
end

@inline function _countPotentials(potentials::Vector{Float64},
  ζcounts::Vector{Int64}, model::Model, rpio::RPIO{Int64Particle}, p::Int64)
  if rpio.threads > 1 && length(ζcounts) > PARALLEL_COUNTS_THRESHOLD
    return _countPotentialsParallel(potentials, ζcounts, rpio, p, model.G)
  else
    return _countPotentialsSerial(potentials, ζcounts, rpio, p, model.G)
  end
end

@inline function _getCountsSerial(rpio::RPIO{Int64Particle})
  particles::Vector{Int64Particle} = rpio.particles
  ζmax::Int64 = maximumI64(particles)
  ζcounts::Vector{Int64} = zeros(Int64, ζmax + 1)
  for i in 1:length(particles)
    @inbounds ζcounts[particles[i].x+1] += 1
  end
  return ζcounts
end

@inline function _threadCounts(threadCounts::Vector{Vector{Int64}},
  particles::Vector{Int64Particle}, ζmax::Int64)
  M::Int64 = length(particles)
  nt::Int64 = length(threadCounts)
  b::Int64 = div(M, nt)
  Threads.@threads for i in 1:nt
    for j in 1:ζmax + 1
      threadCounts[i][j] = 0
    end
    start::Int64 = (i-1)*b + 1
    finish::Int64 = i != nt ? i*b : M
    for j in start:finish
      @inbounds threadCounts[i][particles[j].x+1] += 1
    end
  end
end

@inline function _sumCounts(ζcounts::Vector{Int64},
  threadCounts::Vector{Vector{Int64}})
  M::Int64 = length(ζcounts)
  nt::Int64 = length(threadCounts)
  b::Int64 = div(M, nt)
  Threads.@threads for i in 1:nt
    start::Int64 = (i-1)*b + 1
    finish::Int64 = i != nt ? i*b : M
    for j in start:finish
      @inbounds ζcounts[j] = 0
      for k in 1:nt
        @inbounds ζcounts[j] += threadCounts[k][j]
      end
    end
  end
end

@inline function _maxParticlesParallel(rpio::RPIO{Int64Particle})
  nt::Int64 = rpio.threads
  particles::Vector{Int64Particle} = rpio.particles
  N::Int64 = rpio.N
  b::Int64 = div(N, nt)
  scratchInt::Vector{Int64} = rpio.scratchInt
  Threads.@threads for i in 1:nt
    start::Int64 = (i-1)*b + 1
    finish::Int64 = i != nt ? i*b : N
    v::Int64 = 0
    for j in start:finish
      @inbounds v = max(v, particles[j].x)
    end
    @inbounds scratchInt[i] = v
  end
  return maximum(scratchInt)
end

@inline function _getCountsParallel(rpio::RPIO{Int64Particle})
  ζmax::Int64 = _maxParticlesParallel(rpio)
  ζcounts::Vector{Int64} = Vector{Int64}(undef, ζmax + 1)
  threadCounts::Vector{Vector{Int64}} =
    Vector{Vector{Int64}}(undef, rpio.threads)
  for i in 1:rpio.threads
    threadCounts[i] = Vector{Int64}(undef, ζmax + 1)
  end
  _threadCounts(threadCounts, rpio.particles, ζmax)
  _sumCounts(ζcounts, threadCounts)
  return ζcounts
end


@inline function _getCounts(rpio::RPIO{Int64Particle})
  if rpio.threads > 1
    return _getCountsParallel(rpio)
  else
    return _getCountsSerial(rpio)
  end
end
