using RecyclePermanent
using RNGPool
import NonUniformRandomVariateGeneration.sampleCategorical
using BenchmarkTools
using Combinatorics

include("compare.jl")

n = 100
N = 1000

A = exp.(randn(n, N)) ./ N

trials = 1000

setRNGs(12345)

function permanent(A::Matrix)
  n, N = size(A)
  perms = permutations(1:N, n)
  v::Float64 = 0.0
  for σ in perms
    u::Float64 = 1.0
    for i in 1:n
      u *= A[i, σ[i]]
    end
    v += u
  end
  return v
end

function kuznetsovLogPermanent(A::Matrix, ks::Vector{Int64},
  probs::Vector{Float64})
  n, N = size(A)
  v::Float64 = 0.0
  for p in 1:n
    for i in 1:N
      @inbounds probs[i] = A[p,i]
    end
    for j in 1:p-1
      @inbounds probs[ks[j]] = 0.0
    end
    sp::Float64 = sum(probs)
    if sp == 0 return -Inf end
    k::Int64 = sampleCategorical(probs, sp)
    v += log(sp)
    p < n && (ks[p] = k)
  end
  return v
end

function kuznetsovLogPermanent(A::Matrix)
  n, N = size(A)
  ks = Vector{Int64}(undef, n-1)
  probs = Vector{Float64}(undef, N)
  return kuznetsovLogPermanent(A, ks, probs)
end

# too slow for large n, N
# lp = log(permanent(A))

kuznetsovOutput = testEstimator(() -> kuznetsovLogPermanent(A), trials)
kmOutput = testEstimator(() -> RecyclePermanent.kmLogPermanent(A), trials)

@btime kuznetsovLogPermanent($A)
@btime RecyclePermanent.kmLogPermanent($A)
