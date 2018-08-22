using Combinatorics
using RecyclePermanent
using RNGPool

setRNGs(12345)
rng = getRNG()

function kmPermanent(A::Matrix, ϵ::Float64, N::Int64)
  v::Float64 = 0.0
  for i in 1:N
    v += exp(RecyclePermanent.kmLogPermanent(A, ϵ))
  end
  return v / N
end

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

A = rand(rng, 5, 5)
v1 = permanent(A)
v2 = kmPermanent(A, 0.0, 1000000)
@test v1 ≈ v2 rtol=0.05

A = rand(rng, 2, 5)
v1 = permanent(A)
v2 = kmPermanent(A, 0.0, 1000000)
@test v1 ≈ v2 rtol=0.05

A = round.(Int64, rand(rng, 5, 10) .- 0.1)
v1 = permanent(A)
v2 = kmPermanent(A, 0.05, 1000000)
@test v1 ≈ v2 rtol=0.05
