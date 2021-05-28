
## Function to simulate multivariate species counts which are
## correlated in space and between species.
## n_p: number of species, n_f: number of factors, n_s: Number of sites
simulate_sfa <- function(n_p, n_s, n_f, SpatialScale=0.1,
                         SD_O=1.0, logMeanDens=3.0){
  ## Loadings matrix
  Loadings_pf <- matrix( rnorm(n_f*n_p), nrow=n_p, ncol=n_f)
  for(fI in 1:n_f) Loadings_pf[seq(from=1,to=fI-1,length=fI-1),fI] = 0
  Beta_p <- rep(logMeanDens, n_p)        # intercepts by species
  ## Random locations for sites
  Loc_xy <- cbind(x=runif(n_stations), y=runif(n_stations))

  ## Initialize spatial model and simulate spatial fields
  model_O <- RMgauss(var=SD_O^2, scale=SpatialScale)
  Omega_sf <- matrix(NA, ncol=n_f, nrow=n_s)
  for(fI in 1:n_f)
    Omega_sf[,fI] <- RFsimulate(model=model_O, x=Loc_xy[,'x'], y=Loc_xy[,'y'])@data[,1]

  ## Calculate linear predictor and simulate counts
  ln_Yexp_sp <- Omega_sf%*%t(Loadings_pf) + outer(rep(1,n_s),Beta_p)
  Y_sp <- matrix(rpois(n_s*n_p, lambda=exp(ln_Yexp_sp)), ncol=n_p, byrow=FALSE)

  simdata <- list(Y_sp=Y_sp, Loadings_pf=Loadings_pf,
                  Loc_xy=Loc_xy, Omega_sf=Omega_sf,
                  ln_Yexp_sp=ln_Yexp_sp)
  return(simdata)
}
