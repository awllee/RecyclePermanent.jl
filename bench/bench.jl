import StatsFuns.logsumexp
import Statistics.var

logmeanexp(a) = logsumexp(a) - log(length(a))

function testEstimator(estimator, trials)
  vals::Vector{Float64} = Vector{Float64}(undef, trials)
  totalTime = @elapsed for trial in 1:trials
    vals[trial] = estimator()
  end
  m::Float64 = logmeanexp(vals)
  v::Float64 = var(exp.(vals .- m))
  println("time per trial = ", totalTime / trials )
  println("log(mean) ≈ ", m, ", relative variance ≈ ", v)
end
