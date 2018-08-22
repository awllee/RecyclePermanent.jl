include("gkMCSetup.jl")

function makeRecycleLTD(ys::Vector{Float64}, N::Int64)
  rpio::RPIO{Float64Particle} = RPIO{Float64Particle}(N, length(ys), 1)
  function ltd(in::SVector{d, Float64})
    θ::GKθ = toGKθ(in)
    lp::Float64 = lglogprior(θ)
    if lp == -Inf return -Inf end
    model::Model = makeGKModel(θ, ys)
    return rpEstimate(model, rpio)
  end
end

logtargetRecycle = makeRecycleLTD(ys, N)
PRecycle = makeAMKernel(logtargetRecycle, d)

Random.seed!(12345)
setRNGs(54321)

recycleElapsed = @elapsed (chainRecycle = simulateChainProgress(PRecycle, t0, chainLength))
rar = PRecycle(:acceptanceRate)
recycleCovEstimate = PRecycle(:covEstimate)

@save "data/gkMCRecycle.jld" chainRecycle rar recycleCovEstimate recycleElapsed
