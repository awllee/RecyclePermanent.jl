import StatsFuns.logsumexp
import Statistics.var

@inline function logmeanexp(array::Vector{Float64})
  return logsumexp(array) - log(length(array))
end

function testEstimator(estimator, trials)
  vals::Vector{Float64} = Vector{Float64}(undef, trials)
  totalTime = @elapsed for trial in 1:trials
    vals[trial] = estimator()
  end
  logMean::Float64 = logmeanexp(vals)
  relativeVariance::Float64 = var(exp.(vals .- logMean))
  averageTime::Float64 = totalTime / trials * Threads.nthreads()
  return averageTime, logMean, relativeVariance
end
