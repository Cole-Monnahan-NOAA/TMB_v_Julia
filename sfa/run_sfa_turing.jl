using Turing
using ReverseDiff; Turing.setadbackend(:forwarddiff)
# using Zygote; Turing.setadbackend(:zygote)
using Memoization
using FactorLoadingMatrices
using GaussianMarkovRandomFields
using SparseArrays
using CSV
using DataFrames
using StatsPlots

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


@model function SpatialFactorModel(y, idx, M0, M1, M2, nfactor, ::Type{T}=Float64) where {T}
    nobs, nspp = size(y)
    nnodes = size(M0, 1)
    ## priors
    # species-specific offsets
    logβ ~ filldist(Normal(0, 5), nspp)
    β = exp.(logβ)
    # Factor loading matrix
    𝓁 ~ filldist(Normal(0, 1), nnz_loading(nspp, nfactor)) # as flat vector
    L = loading_matrix(𝓁, nspp, nfactor) # reshape into nspp x nfactor matrix
    # spatial autocorrelation structure
    logκ ~ Normal(0, 5) # SPDE range parameter
    κ = exp(logκ)
    τ = 1 / (κ * sqrt(4π))
    range = sqrt(8) / κ
    Q = κ^4 * M0 + 2κ^2 * M1 + M2

    # random spatial field
    Ω = Matrix{T}(undef, nnodes, nfactor)
    G = GMRF(Q)
    for i in 1:nfactor
        Ω[:, i] ~ G
    end
    λ = exp.(Ω * L' .+ β')
    y ~ arraydist(Poisson.(λ[idx, :]))
end

# Instantiate model and fit using variational inference
sfmod = SpatialFactorModel(y, idx, Matrix(M0), Matrix(M1), Matrix(M2), 5)
res = vi(sfmod, ADVI(3, 1000)) # 2.5 h

# extract fitted parameters
ilogβ = 1:size(y, 2)
i𝓁 = (1:nnz_loading(size(y,2), 5)) .+ maximum(ilogβ)
ilogκ = maximum(i𝓁) + 1
iΩ = ilogκ+1:length(res.dist.m)
β = res.dist.m[ilogβ]
Ω = reshape(res.dist.m[iΩ], :, 5)
L = loading_matrix(res.dist.m[i𝓁], 10, 5)
κ = exp(res.dist.m[ilogκ])

# read in true values for the parameters as simulated in R
Ω_true = Array(CSV.read("sfa/Omega.csv", DataFrame)[:, 2:end])
L_true = Array(CSV.read("sfa/Loadings.csv", DataFrame)[:, 2:end])

plot(heatmap(L), heatmap(L_true[:, 1:5]), clim=(-2, 2))

plot(exp.(Ω * L' .+ β'), layout=(5, 2), size=(1200, 1000));
plot!(exp.(Ω_true * L_true' .+ β'))

β_tmb = [2.83872322, 2.92016159, 3.16893578, 2.87994756, 3.60857897, 2.93218035, 3.57939559,
    2.87064375, 3.41632312, 2.93218011]
plot(β_tmb)
plot!(β)


MvNormal(res.dist.m[ilogβ], res.dist.σ[ilogβ])
MvNormal(res.dist.m[i𝓁], res.dist.σ[i𝓁])
Normal(res.dist.m[ilogκ], res.dist.σ[ilogκ])
MvNormal(res.dist.m[iΩ], res.dist.σ[iΩ])


Turing.setadbackend(:forwarddiff)
chain = sample(sfmod, NUTS(), 10)
