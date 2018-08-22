include("gkMCSetup.jl")

function makeBiasedLTD(ys::Vector{Float64}, N::Int64)
  rpio::RPIO{Float64Particle} = RPIO{Float64Particle}(N, length(ys), 1)
  function ltd(in::SVector{d, Float64})
    θ::GKθ = toGKθ(in)
    lp::Float64 = lglogprior(θ)
    if lp == -Inf return -Inf end
    model::Model = makeGKModel(θ, ys)
    return biasedEstimate(model, rpio)
  end
end

logtargetBiased = makeBiasedLTD(ys, N)
PBiased = makeAMKernel(logtargetBiased, d)

Random.seed!(12345)
setRNGs(54321)

biasedElapsed = @elapsed (chainBiased = simulateChainProgress(PBiased, t0, chainLength))
bar = PBiased(:acceptanceRate)
biasedCovEstimate = PBiased(:covEstimate)

@save "data/gkMCBiased.jld" chainBiased bar biasedCovEstimate biasedElapsed
