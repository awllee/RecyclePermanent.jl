using RecyclePermanent
using RNGPool
import Statistics: mean, var
import NonUniformRandomVariateGeneration.sampleCategorical
import LinearAlgebra: mul!, det
using DataFrames

function kuznetsovLogPermanent(A::Matrix, ks::Vector{Int64}, probs::Vector{Float64})
  N, n = size(A)
  @assert N >= n
  rng = getRNG()
  v::Float64 = 0.0
  for p in 1:n
    for i in 1:N
      @inbounds probs[i] = A[i, p]
    end
    for j in 1:p-1
      @inbounds probs[ks[j]] = 0.0
    end
    sp::Float64 = sum(probs)
    if sp == 0 return -Inf end
    k::Int64 = sampleCategorical(probs, sp, rng)
    v += log(sp)
    p < n && (ks[p] = k)
  end
  return v
end

function _signedBernoulli(rng)
  return 2*(rand(rng) < 0.5) - 1
end

function GodsilGutman(A::Matrix, X::Matrix, Z::Matrix)
  m, n = size(A)
  rng = getRNG()
  for i in 1:m
    for j in 1:n
      X[i,j] = sqrt(A[i,j]) * _signedBernoulli(rng)
    end
  end
  mul!(Z, X', X)
  return det(Z)
end

function ggPermanent(A::Matrix, N::Int64)
  m, n = size(A)
  @assert m >= n
  vs::Vector{Float64} = Vector{Float64}(undef, N)
  X = Matrix{Float64}(undef, m, n)
  Z = Matrix{Float64}(undef, n, n)
  for i in 1:N
    vs[i] = GodsilGutman(A, X, Z)
  end
  return vs
end

function kmPermanent(A::Matrix, ϵ::Float64, N::Int64)
  m, n = size(A)
  @assert m >= n
  vs::Vector{Float64} = Vector{Float64}(undef, N)
  ks = Vector{Int64}(undef, n-1)
  Cs = Vector{Float64}(undef, m)
  probs = Vector{Float64}(undef, m)
  for i in 1:N
    vs[i] = exp(RecyclePermanent.kmLogPermanent(A, ks, Cs, probs, ϵ))
  end
  return vs
end

function kuznetsovPermanent(A::Matrix, N::Int64)
  m, n = size(A)
  @assert m >= n
  vs::Vector{Float64} = Vector{Float64}(undef, N)
  ks = Vector{Int64}(undef, n-1)
  probs = Vector{Float64}(undef, m)
  for i in 1:N
    vs[i] = exp(kuznetsovLogPermanent(A, ks, probs))
  end
  return vs
end

function makeBinaryMatrix(m, n)
  rng = getRNG()
  c = 1.0
  for p = 1:n
    c *= m/(m-p+1)
  end
  return (rand(rng, m, n) .< 0.5) .* (2/m * c^(1/n))
end

function experiment(m, n, trials)
  A = makeBinaryMatrix(m, n)
  x = maximum(A)
  multipliers = [1000000, 10000, 100, 1, .1]
  t = @elapsed (vs = kuznetsovPermanent(A, trials))
  df = DataFrame(Algorithm = "K", TimeMS = t/trials*1000, Mean = mean(vs), Variance = var(vs))
  t = @elapsed (vs = ggPermanent(A, trials))
  push!(df, ["GG", t/trials*1000, mean(vs), var(vs)])
  for multiplier in multipliers
    mytrials = multiplier >= 1 ? trials : trials*10
    t = @elapsed (vs = kmPermanent(A, x*multiplier, mytrials))
    push!(df, ["KM: "*string(multiplier), t/mytrials*1000, mean(vs), var(vs)])
  end
  println("Experiment: m = ", m, ", n = ", n)
  show(df)
  println()
end

setRNGs(12345)

experiment(1000, 10, 10000)
experiment(1000, 100, 10000)
