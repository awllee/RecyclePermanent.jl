include("../models/bernoulliModel.jl")

setRNGs(12345)

θ = 0.4

model = makeBernoulliModel(θ)
ys = RecyclePermanent.simulateModel(model, 100)
model = makeBernoulliModel(θ, ys)

N = 1000*model.n

trueValue = sum(ys)*log(θ)+(model.n-sum(ys))*log(1-θ)

kmio = KMIO{model.Tx}(N, model.n)
@test kmEstimate(model, kmio) ≈ trueValue atol=0.1

rpio = RPIO{model.Tx}(N, model.n, 1)
@test simpleEstimate(model, rpio) ≈ trueValue atol=0.1
@test biasedEstimate(model, rpio) ≈ trueValue atol=0.1
@test biasedEstimateDiscrete(model, rpio) ≈ trueValue atol=0.1
@test rpEstimate(model, rpio) ≈ trueValue atol=0.1
@test rpEstimateDiscrete(model, rpio) ≈ trueValue atol=0.1

rpio = RPIO{model.Tx}(N, model.n, Threads.nthreads())
@test simpleEstimate(model, rpio) ≈ trueValue atol=0.1
@test biasedEstimate(model, rpio) ≈ trueValue atol=0.1
@test biasedEstimateDiscrete(model, rpio) ≈ trueValue atol=0.1
@test rpEstimate(model, rpio) ≈ trueValue atol=0.1
@test rpEstimateDiscrete(model, rpio) ≈ trueValue atol=0.1

N = 3*model.n
trials = 10000
vals = Vector{Float64}(undef, trials)

t1 = 1+(θ/θ^2-1)/N
t2 = 1+((1-θ)/(1-θ)^2-1)/N
trueVar = t1^sum(ys)*t2^(model.n - sum(ys)) - 1

rpio = RPIO{model.Tx}(N, model.n, Threads.nthreads())

for i in 1:trials
  vals[i] = exp(simpleEstimate(model, rpio) - trueValue)
end
@test mean(vals) ≈ 1.0 atol=0.1
@test var(vals) ≈ trueVar atol=0.1

for i in 1:trials
  vals[i] = exp(rpEstimate(model, rpio) - trueValue)
end
@test mean(vals) ≈ 1.0 atol=0.1

for i in 1:trials
  vals[i] = exp(rpEstimateDiscrete(model, rpio) - trueValue)
end
@test mean(vals) ≈ 1.0 atol=0.1

kmio = KMIO{model.Tx}(N, model.n)
for i in 1:trials
  vals[i] = exp(kmEstimate(model, kmio) - trueValue)
end
@test mean(vals) ≈ 1.0 atol=0.1
