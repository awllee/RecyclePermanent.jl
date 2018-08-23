using RecyclePermanent
using RNGPool
using JLD2
import Statistics: mean, var

include("../models/gkModel.jl")

@load "data/gkData.jld" ys θ
ys = ys[1:100]
model = makeGKModel(θ, ys)

setRNGs(12345)

N = 100000*model.n
rpio = RPIO{model.Tx}(N, model.n, 1)
oracle = rpEstimate(model, rpio)

trials = 10000
cs = 50:10:100

means1 = Vector{Float64}(undef, length(cs))
vars1 = Vector{Float64}(undef, length(cs))
means2 = Vector{Float64}(undef, length(cs))
vars2 = Vector{Float64}(undef, length(cs))

for k in 1:length(cs)
  println(cs[k])
  N = cs[k]*model.n
  rpio = RPIO{model.Tx}(N, model.n, 1)
  vs1 = Vector{Float64}(undef, trials)
  vs2 = Vector{Float64}(undef, trials)
  for i in 1:trials
    vs1[i] = rpEstimate(model, rpio)
    vs2[i] = biasedEstimate(model, rpio)
  end
  means1[k] = mean(exp.(vs1 .- oracle))
  vars1[k] = var(exp.(vs1 .- oracle))
  means2[k] = mean(exp.(vs2 .- oracle))
  vars2[k] = var(exp.(vs2 .- oracle))
end

println(means1)
println(vars1)
println(means2)
println(vars2)
println((means2 .- 1).^2 .+ vars2)
