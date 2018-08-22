using RecyclePermanent
using RNGPool
using JLD2
using Random
import Statistics: mean, var
using Test

@testset "Bernoulli test" begin
  @time include("bernoulli_test.jl")
end

@testset "BP test" begin
  @time include("bp_test.jl")
end

@testset "Kou--McCullagh test" begin
  @time include("km_test.jl")
end
