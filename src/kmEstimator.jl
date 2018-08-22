import NonUniformRandomVariateGeneration.sampleCategorical

struct KMIO{Particle}
  N::Int64
  n::Int64
  particles::Vector{Particle}
  potentials::Array{Float64, 2}
  ks::Vector{Int64}
  Cs::Vector{Float64}
  probs::Vector{Float64}
end

function KMIO{Particle}(N::Int64, n::Int64) where Particle
  particles::Vector{Particle} = Vector{Particle}(undef, N)
  for i in 1:N
    particles[i] = Particle()
  end
  potentials::Array{Float64, 2} = Matrix{Float64}(undef, N, n)
  ks::Vector{Int64} = Vector{Int64}(undef, n-1)
  Cs::Vector{Float64} = Vector{Float64}(undef, N)
  probs::Vector{Float64} = Vector{Float64}(undef, N)
  return KMIO{Particle}(N, n, particles, potentials, ks, Cs, probs)
end

@inline function _makePotentialMatrix(μ!::F1, G::F2,
  kmio::KMIO{Particle}) where {F1<:Function, F2<:Function, Particle}
  n::Int64 = kmio.n
  N::Int64 = kmio.N
  particles::Vector{Particle} = kmio.particles
  potentials::Array{Float64, 2} = kmio.potentials
  rng = getRNG()
  for i in 1:N
    @inbounds μ!(particles[i], rng)
  end
  C::Float64 = 1/N
  for p in 1:n
    for i in 1:N
      @inbounds kmio.potentials[i, p] = G(p, particles[i]) * C
    end
  end
end

@inline function kmLogPermanent(A::Matrix, ks::Vector{Int64}, Cs::Vector{Float64},
  probs::Vector{Float64}, ϵ::Float64)
  N, n = size(A)
  @assert N >= n
  @assert N == length(Cs)
  for i in 1:N
    @inbounds Cs[i] = 0.0
    for p in 1:n
      @inbounds Cs[i] += A[i, p]
    end
  end
  rng = getRNG()
  v::Float64 = 0.0
  for p in 1:n
    if p < n
      for i in 1:N
        @inbounds if A[i, p] > 0
          @inbounds probs[i] = A[i, p] / (Cs[i] - A[i, p] + ϵ)
        else
          @inbounds probs[i] = 0.0
        end
      end
    else
      for i in 1:N
        @inbounds probs[i] = A[i, p]
      end
    end
    for j in 1:p-1
      @inbounds probs[ks[j]] = 0.0
    end
    sp::Float64 = sum(probs)
    if sp == 0 return -Inf end
    k::Int64 = sampleCategorical(probs, sp, rng)
    for i in 1:N
      @inbounds Cs[i] -= A[i, p]
    end
    v += log(A[k, p] * sp / probs[k])
    p < n && (ks[p] = k)
  end
  return v
end

function kmLogPermanent(A::Matrix, ϵ::Float64 = 0.0)
  N, n = size(A)
  if N < n
    At = Matrix(A')
    return kmLogPermanent(At, ϵ)
  end
  ks = Vector{Int64}(undef, n-1)
  Cs = Vector{Float64}(undef, N)
  probs = Vector{Float64}(undef, N)
  return kmLogPermanent(A, ks, Cs, probs, ϵ)
end

@inline function kmEstimate(model::Model, kmio::KMIO{Particle},
  ϵ::Float64 = 0.0) where Particle
  _makePotentialMatrix(model.μ!, model.G, kmio)
  v::Float64 = kmLogPermanent(kmio.potentials, kmio.ks, kmio.Cs, kmio.probs, ϵ)
  n::Int64 = kmio.n
  N::Int64 = kmio.N
  lPNn::Float64 = 0.0
  for p in 1:n
    lPNn += log(N-p+1)
  end
  return v - lPNn + n*log(N)
end
