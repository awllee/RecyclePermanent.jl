include("bpMCSetup.jl")

import MonteCarloMarkovKernels: estimateBM, kde
using StatsBase
import Statistics.var

using Plots
Plots.gr()

@load "data/bpMCSimple.jld" chainSimple sar simpleCovEstimate simpleElapsed
@load "data/bpMCRecycle.jld" chainRecycle rar recycleCovEstimate recycleElapsed
@load "data/bpMCBiased.jld" chainBiased bar biasedCovEstimate biasedElapsed

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
savefigures && savefig("bp_kde3.png")

contour(kde(vsSimple[1], vsSimple[2], sar))
contour!(kde(vsRecycle[1], vsRecycle[2], rar))
savefigures && savefig("bp_kde12.png")

contour(kde(vsSimple[1], vsSimple[3], sar))
contour!(kde(vsRecycle[1], vsRecycle[3], rar))
savefigures && savefig("bp_kde13.png")
#
contour(kde(vsSimple[2], vsSimple[3], sar))
contour!(kde(vsRecycle[2], vsRecycle[3], rar))
savefigures && savefig("bp_kde23.png")

plot(autocor(vsSimple[1]))
plot!(autocor(vsRecycle[1]))
savefigures && savefig("pmmh_acf1.png")
plot(autocor(vsSimple[2]))
plot!(autocor(vsRecycle[2]))
savefigures && savefig("pmmh_acf2.png")
plot(autocor(vsSimple[3]))
plot!(autocor(vsRecycle[3]))
savefigures && savefig("pmmh_acf3.png")

avarsSimple = estimateBM.(vsSimple)
avarsRecycle = estimateBM.(vsRecycle)
avarsBiased = estimateBM.(vsBiased)

varsSimple = var.(vsSimple)
varsRecycle = var.(vsRecycle)
varsBiased = var.(vsBiased)

essesSimple = varsSimple./avarsSimple*chainLength
essesRecycle = varsRecycle./avarsRecycle*chainLength
essesBiased = varsBiased./avarsBiased*chainLength

estimatedSimpleElapsed = 4536000

essPerSecondSimple = essesSimple / estimatedSimpleElapsed
essPerSecondRecycle = essesRecycle / recycleElapsed
essPerSecondBiased = essesBiased / biasedElapsed

plot(kde(vsRecycle[1], rar), label="", xlabel="lambda", ylabel="density")
savefig("demo/bp_kde1.pdf")

plot(kde(vsRecycle[2], rar), label="", xlabel="k_on", ylabel="density")
savefig("demo/bp_kde2.pdf")

plot(kde(vsRecycle[3], rar), label="", xlabel="k_off", ylabel="density")
savefig("demo/bp_kde3.pdf")
