module RecyclePermanent

using RNGPool
using Random

include("model.jl")
include("rpio.jl")
include("common.jl")
include("commonDiscrete.jl")
include("simpleEstimator.jl")
include("biasedEstimator.jl")
include("rpEstimator.jl")
include("kmEstimator.jl")

export Model, simulateModel, RPIO, simpleEstimate, biasedEstimate, rpEstimate,
  rpEstimateDiscrete, biasedEstimateDiscrete, Int64Particle, KMIO, kmEstimate

end # module
