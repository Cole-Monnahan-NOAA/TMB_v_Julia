using SparseArrays, LinearAlgebra
using DataFrames, CSV
using FactorLoadingMatrices
using Distributions
using ForwardDiff, ReverseDiff, Zygote
using GaussianMarkovRandomFields
using TransformVariables
using MarginalLogDensities
using StatsPlots
using BenchmarkTools


struct SpatialFactorProblem{TM,I}
    y::Matrix{I}
    idx::Vector{I}
    M0::TM
    M1::TM
    M2::TM
    nfactors::I
end
nobservations(sfp::SpatialFactorProblem) = size(sfp.y, 1)
nspecies(sfp::SpatialFactorProblem) = size(sfp.y, 2)
nfactors(sfp::SpatialFactorProblem) = sfp.nfactors
nnodes(sfp::SpatialFactorProblem) = size(sfp.M0, 1)

function parameter_transformation(sfp::SpatialFactorProblem)
    nspp = nspecies(sfp)
    nfact = nfactors(sfp)
    nn = nnodes(sfp)
    return as((κ = as_positive_real,
        β = as(Array, nspp),
        𝓁 = as(Array, nnz_loading(nspp, nfact)),
        Ω = as(Array, nn, nfact)))
end

function loglik(sfp::SpatialFactorProblem, θ::Vector)
    nobs = nobservations(sfp)
    nspp = nspecies(sfp)
    nfact = nfactors(sfp)
    nn = nnodes(sfp)
    # unpack parameters
    trans = parameter_transformation(sfp)
    κ, β, 𝓁, Ω = TransformVariables.transform(trans, θ)

    τ = 1.0 / (κ * sqrt(4π))
    Q = τ^2 * (κ^4 * sfp.M0 + 2κ^2 * sfp.M1 + sfp.M2)

    L = loading_matrix(𝓁, nspp, nfact)
    λ = exp.(Ω * L' .+ β')[sfp.idx, :]
    G = GMRF(Q)

    return sum([logpdf(G, Ω[:, i]) for i in 1:nfact]) + sum(logpdf.(Poisson.(λ), sfp.y))
end

# Reading in the model inputs we saved from the R script
df = CSV.read(joinpath(@__DIR__, "counts.csv"), DataFrame)

y = Matrix(df[:, Symbol.("X" .* string.(1:10))])
idx = df.idx .+ 1
M0_df = CSV.read(joinpath(@__DIR__, "M0.csv"), DataFrame)
M1_df = CSV.read(joinpath(@__DIR__, "M1.csv"), DataFrame)
M2_df = CSV.read(joinpath(@__DIR__, "M2.csv"), DataFrame)

M0 = (sparse(M0_df.i .+ 1, M0_df.j .+ 1, M0_df.x))
M1 = (sparse(M1_df.i .+ 1, M1_df.j .+ 1, M1_df.x))
M2 = (sparse(M2_df.i .+ 1, M2_df.j .+ 1, M2_df.x))

sfp = SpatialFactorProblem(y, idx, M0, M1, M2, 2)
trans = parameter_transformation(sfp)

θ = zeros(TransformVariables.dimension(trans))
@btime loglik($sfp, $θ) # 613 μs
@code_warntype loglik(sfp, θ)
@profview for _ in 1:1000
    loglik(sfp, θ)
end

trans_dims = [TransformVariables.dimension.(values(trans.transformations))...]
trans_offsets = cumsum([0; trans_dims[1:end-1]])
nθ = TransformVariables.dimension(trans)


mld = MarginalLogDensity(θ -> loglik(sfp, θ), nθ, collect(31:nθ), forwarddiff_sparsity=true)
njoint(mld)
heatmap(Matrix(mld.hessconfig.Hsparsity), yflip=true)

β = [4.044717, 3.323056, 3.592093, 5.732451, 5.789991, 4.382527, 4.710431, 6.519988,
    4.963544, 7.092935]
κ = exp(0.0)
𝓁 = ones(19)
const θjoint = [log(κ); β; 𝓁]

u = randn(nmarginal(mld))
@code_warntype mld(u, θjoint)
@profiler mld(θjoint, true)
MarginalLogDensities._marginalize(mld, θjoint, LaplaceApprox(), true)

##
using FiniteDiff

function loglik1(sfp::SpatialFactorProblem, u, θ)
    nobs = nobservations(sfp)
    nspp = nspecies(sfp)
    nfact = nfactors(sfp)
    nn = nnodes(sfp)
    # unpack parameters
    trans = parameter_transformation(sfp)
    Ω = reshape(u, nn, nfact)
    κ = exp(θ[1])
    𝓁 = θ[2:nnz_loading(nspp, nfact)+1]
    β = θ[nnz_loading(nspp, nfact)+2:end]

    τ = 1.0 / (κ * sqrt(4π))
    Q = τ^2 * (κ^4 * sfp.M0 + 2κ^2 * sfp.M1 + sfp.M2)

    L = loading_matrix(𝓁, nspp, nfact)
    λ = exp.(Ω * L' .+ β')[sfp.idx, :]
    G = GMRF(Q)

    return sum([logpdf(G, Ω[:, i]) for i in 1:nfact]) + sum(logpdf.(Poisson.(λ), sfp.y))
end

const s = sfp
u = randn(432)
@code_warntype loglik1(sfp, u, θjoint)
# 1
g(v) = -loglik1(s, v[1:432], v[433:end])

v = randn(462)
@code_warntype g(v)

# 2
gconf = ForwardDiff.GradientConfig(g, v)
∇ᵥg(v::T) where T = ForwardDiff.gradient(g, v)::T
∇ᵥg!(G, v) = ForwardDiff.gradient!(G, g, v)
@code_warntype ∇ᵥg(v)
∇ᵥg(v)
@btime g(v)
@btime ∇ᵥg(v)

G = zeros(462)
ForwardDiff.gradient!(G, g, v, gconf)
∇ᵥg!(G, v)
@code_warntype ∇ᵥg!(G, v)
@btime ∇ᵥg!(G, v)


# 3
𝐇(v) = ReverseDiff.jacobian(∇ᵥg, v)
𝐇(v)
