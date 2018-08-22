using RecyclePermanent
using RNGPool
using JLD2

include("../models/gkModel.jl")
include("compare.jl")

@load "data/gkData.jld" ys θ
model = makeGKModel(θ, ys)

function findN(f::F, N0::Int64) where F<:Function
  c = 2
  N = c*N0
  output = f(N)
  while isnan(output) || output > 1.5
    c *= 2
    N = c*N0
    output = f(N)
  end
  d = 1.0
  prevOutput = output
  prevN = N
  while !isnan(output) && output < 1.5
    d *= 1.15
    prevOutput = output
    prevN = N
    N = ceil(Int64, c*N0/d)
    output = f(N)
  end
  return prevN, prevOutput
end

function makeRecycleN(model, n::Int64, trials::Int64)
  function recyclef(N::Int64)
    nt = Threads.nthreads()
    rpios = Vector{RPIO{model.Tx}}(undef, nt)
    println(N)
    for i in 1:nt
      rpios[i] = RPIO{model.Tx}(N, n, 1)
    end
    vals::Vector{Float64} = Vector{Float64}(undef, trials)
    Threads.@threads for trial in 1:trials
      vals[trial] = rpEstimate(model, rpios[Threads.threadid()])
    end
    logMean::Float64 = logmeanexp(vals)
    relativeVariance::Float64 = var(exp.(vals .- logMean))
    return relativeVariance
  end
  return recyclef
end

function makeSimpleN(model, n::Int64, trials::Int64)
  function simplef(N::Int64)
    nt = Threads.nthreads()
    rpios = Vector{RPIO{model.Tx}}(undef, nt)
    println(N)
    for i in 1:nt
      rpios[i] = RPIO{model.Tx}(N, n, 1)
    end
    vals::Vector{Float64} = Vector{Float64}(undef, trials)
    Threads.@threads for trial in 1:trials
      vals[trial] = simpleEstimate(model, rpios[Threads.threadid()])
    end
    logMean::Float64 = logmeanexp(vals)
    relativeVariance::Float64 = var(exp.(vals .- logMean))
    return relativeVariance
  end
  return simplef
end

function makeKMN(model, n::Int64, ϵ::Float64, trials::Int64)
  function kmf(N::Int64)
    nt = Threads.nthreads()
    kmios = Vector{KMIO{model.Tx}}(undef, nt)
    println(N)
    for i in 1:nt
      kmios[i] = KMIO{model.Tx}(N, n)
    end
    vals::Vector{Float64} = Vector{Float64}(undef, trials)
    Threads.@threads for trial in 1:trials
      vals[trial] = kmEstimate(model, kmios[Threads.threadid()], ϵ)
    end
    logMean::Float64 = logmeanexp(vals)
    relativeVariance::Float64 = var(exp.(vals .- logMean))
    return relativeVariance
  end
  return kmf
end

trials = 1000

setRNGs(12345)

ns = 50:50:250
outR = Vector{Tuple{Int64, Float64}}(undef, length(ns))
@time for i in 1:length(ns)
  output = findN(makeRecycleN(model, ns[i], trials), ns[i])
  outR[i] = output
end

outS = Vector{Tuple{Int64, Float64}}(undef, length(ns))
@time for i in 1:length(ns)
  output = findN(makeSimpleN(model, ns[i], trials), ns[i])
  outS[i] = output
end

outKM1 = Vector{Tuple{Int64, Float64}}(undef, length(ns))
@time for i in 1:length(ns)
  output = findN(makeKMN(model, ns[i], 1.0, trials), ns[i])
  outKM1[i] = output
end

outKM2 = Vector{Tuple{Int64, Float64}}(undef, length(ns))
@time for i in 1:length(ns)
  output = findN(makeKMN(model, ns[i], 0.001, trials), ns[i])
  outKM2[i] = output
end

outKM3 = Vector{Tuple{Int64, Float64}}(undef, length(ns))
@time for i in 1:length(ns)
  output = findN(makeKMN(model, ns[i], 0.0001, trials), ns[i])
  outKM3[i] = output
end

@save "compare/gktau.jld" ns trials outR outS outKM1 outKM2 outKM3
