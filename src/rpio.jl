mutable struct Int64Particle
  x::Int64
  Int64Particle() = new()
end

struct RPIO{Particle}
  N::Int64
  n::Int64
  threads::Int64
  potentials::Vector{Float64}
  particles::Vector{Particle}
  ks::Vector{Int64}
  scratch::Vector{Float64}
  scratchInt::Vector{Int64}
  scratchInt64Particle::Vector{Int64Particle}
end

function RPIO{Particle}(N::Int64, n::Int64, threads::Int64) where Particle
  potentials::Vector{Float64} = Vector{Float64}(undef, N)
  particles::Vector{Particle} = Vector{Particle}(undef, N)
  for i in 1:N
    particles[i] = Particle()
  end
  ks::Vector{Int64} = Vector{Int64}(undef, n-1)
  scratch::Vector{Float64} = Vector{Float64}(undef, threads)
  scratchInt::Vector{Int64} = Vector{Int64}(undef, threads)
  scratchInt64Particle::Vector{Int64Particle} =
    Vector{Int64Particle}(undef, threads)
  for i in 1:threads
    scratchInt64Particle[i] = Int64Particle()
  end
  return RPIO{Particle}(N, n, threads, potentials, particles, ks, scratch,
    scratchInt, scratchInt64Particle)
end
