struct Model{F1<:Function,F2<:Function,F3<:Function}
  μ!::F1
  G::F2
  n::Int64
  simulate::F3
  Tx::Type
  Ty::Type
end

function simulateModel(model::Model, n::Int64)
  ys = Vector{model.Ty}(undef, n)
  rng = getRNG()
  x::model.Tx = model.Tx()
  for p in 1:n
    model.μ!(x, rng)
    ys[p] = model.simulate(x, rng)
  end
  return ys
end
