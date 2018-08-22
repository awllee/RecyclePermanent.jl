include("gkMCSetup.jl")

function makeSimpleLTD(ys::Vector{Float64}, N::Int64)
  rpio::RPIO{Float64Particle} = RPIO{Float64Particle}(N, length(ys), 1)
  function ltd(in::SVector{d, Float64})
    θ::GKθ = toGKθ(in)
    lp::Float64 = lglogprior(θ)
    if lp == -Inf return -Inf end
    model::Model = makeGKModel(θ, ys)
    return simpleEstimate(model, rpio)
  end
end

logtargetSimple = makeSimpleLTD(ys, N)
PSimple = makeAMKernel(logtargetSimple, d)

Random.seed!(12345)
setRNGs(54321)

simpleElapsed = @elapsed (chainSimple = simulateChainProgress(PSimple, t0, chainLength))
sar = PSimple(:acceptanceRate)
simpleCovEstimate = PSimple(:covEstimate)

@save "data/gkMCSimple.jld" chainSimple sar simpleCovEstimate simpleElapsed
