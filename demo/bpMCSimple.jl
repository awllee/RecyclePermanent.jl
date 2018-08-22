include("bpMCSetup.jl")

function makeSimpleLTD(ys::Vector{Float64}, N::Int64)
  rpio::RPIO{Int64Particle} = RPIO{Int64Particle}(N, length(ys),
    Threads.nthreads())
  function ltd(in::SVector{d, Float64})
    θ::BPθ = toBPθ(in)
    lp::Float64 = lglogprior(θ)
    if lp == -Inf return -Inf end
    model::Model = makeBPModel(θ, ys)
    return lp + simpleEstimate(model, rpio)
  end
end

logtargetSimple = makeSimpleLTD(ys, N)
PSimple = makeAMKernel(logtargetSimple, d)

Random.seed!(12345)
setRNGs(54321)

simpleElapsed = @elapsed (chainSimple = simulateChainProgress(PSimple, t0, chainLength))
sar = PSimple(:acceptanceRate)
simpleCovEstimate = PSimple(:covEstimate)

@save "data/bpMCSimple.jld" chainSimple sar simpleCovEstimate simpleElapsed
