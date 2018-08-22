include("gkMCSetup.jl")

import MonteCarloMarkovKernels: estimateBM, kde
using StatsBase
using Plots
import Statistics.var

Plots.gr()

@load "data/gkMCSimple.jld" chainSimple sar simpleCovEstimate simpleElapsed
@load "data/gkMCRecycle.jld" chainRecycle rar recycleCovEstimate recycleElapsed
@load "data/gkMCBiased.jld" chainBiased bar biasedCovEstimate biasedElapsed

savefigures = false

vsSimple = (i->(x->x[i]).(chainSimple)).(1:d)
vsRecycle = (i->(x->x[i]).(chainRecycle)).(1:d)
vsBiased = (i->(x->x[i]).(chainBiased)).(1:d)

plot(kde(vsSimple[1], sar))
plot!(kde(vsRecycle[1], rar))
plot!(kde(vsBiased[1], bar))
savefigures && savefig("gk_kde1.png")

plot(kde(vsSimple[2], sar))
plot!(kde(vsRecycle[2], rar))
plot!(kde(vsBiased[2], bar))
savefigures && savefig("gk_kde2.png")

plot(kde(vsSimple[3], sar))
plot!(kde(vsRecycle[3], rar))
plot!(kde(vsBiased[3], bar))
savefigures && savefig("gk_kde3.png")

plot(kde(vsSimple[4], sar))
plot!(kde(vsRecycle[4], rar))
plot!(kde(vsBiased[4], bar))
savefigures && savefig("gk_kde4.png")

contour(kde(vsSimple[1], vsSimple[2], sar))
contour!(kde(vsRecycle[1], vsRecycle[2], rar))
savefigures && savefig("gk_kde12.png")

contour(kde(vsSimple[1], vsSimple[3], sar))
contour!(kde(vsRecycle[1], vsRecycle[3], rar))
savefigures && savefig("gk_kde13.png")

contour(kde(vsSimple[2], vsSimple[3], sar))
contour!(kde(vsRecycle[2], vsRecycle[3], rar))
savefigures && savefig("gk_kde23.png")

plot(autocor(vsSimple[1]))
plot!(autocor(vsRecycle[1]))
savefigures && savefig("pmmh_acf1.png")
plot(autocor(vsSimple[2]))
plot!(autocor(vsRecycle[2]))
savefigures && savefig("pmmh_acf2.png")
plot(autocor(vsSimple[3]))
plot!(autocor(vsRecycle[3]))
savefigures && savefig("pmmh_acf3.png")
plot(autocor(vsSimple[4]))
plot!(autocor(vsRecycle[4]))
savefigures && savefig("pmmh_acf4.png")

avarsSimple = estimateBM.(vsSimple)
avarsRecycle = estimateBM.(vsRecycle)
avarsBiased = estimateBM.(vsBiased)

varsSimple = var.(vsSimple)
varsRecycle = var.(vsRecycle)
varsBiased = var.(vsBiased)

essesSimple = varsSimple./avarsSimple*chainLength
essesRecycle = varsRecycle./avarsRecycle*chainLength
essesBiased = varsBiased./avarsBiased*chainLength

essPerSecondSimple = essesSimple / simpleElapsed
essPerSecondRecycle = essesRecycle / recycleElapsed
essPerSecondBiased = essesBiased / biasedElapsed

plot(kde(vsRecycle[1], rar), label="", xlabel="A", ylabel="density")
savefig("demo/gk_kde1.pdf")

plot(kde(vsRecycle[2], rar), label="", xlabel="B", ylabel="density")
savefig("demo/gk_kde2.pdf")

plot(kde(vsRecycle[3], rar), label="", xlabel="g", ylabel="density")
savefig("demo/gk_kde3.pdf")

plot(kde(vsRecycle[4], rar), label="", xlabel="k", ylabel="density")
savefig("demo/gk_kde4.pdf")
