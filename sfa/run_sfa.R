### Example modified with permission from
## https://github.com/James-Thorson/2018_FSH556/tree/master/Week%208%20--%20Multivariate%20models/Lecture%208

### Spatial factor analysis in TMB, to be converted to Julia for
### comparison

## Load libraries
library(TMB)
library(INLA)
library(RandomFields)

## Load a simulator function
source("sfa/sfa_simulator.R" )

## Simulation settings
n_factors_true <- 7
n_factors_estimation <- 5
n_species <- 10
## Scale this from say 50 to 1000 to adjust spatial resolution
## and thus the number of data points but also the number of
## sites
n_stations <- 200

## Simulate data and prepare inputs for TMB
set.seed(2315112)
simdat <- simulate_sfa(n_p=n_species, n_s=n_stations, n_f=n_factors_true)
Y_sp <- simdat$Y_sp      # Poisson counts by (s)ite and s(p)ecies
## Create SPDE mesh using INLA functions
mesh <- inla.mesh.create(simdat$Loc_xy)
spde <- inla.spde2.matern(mesh)
Data <- list(Y_sp=Y_sp, n_f=n_factors_estimation, n_x=mesh$n,
             x_s=mesh$idx$loc-1, # sites indexed from 0 in TMB
             ## The sparse INLA M matrices
             M0=spde$param.inla$M0,
             M1=spde$param.inla$M1,
             M2=spde$param.inla$M2)


# Save relevant data to read into Julia
df <- data.frame(Data$Y_sp)
df$idx <- Data$x_s
write.csv(df, "sfa/counts.csv")
write.csv(data.frame(i = Data$M0@i, j = Data$M0@j, x = Data$M0@x), "sfa/M0.csv")
write.csv(data.frame(i = Data$M1@i, j = Data$M1@j, x = Data$M1@x), "sfa/M1.csv")
write.csv(data.frame(i = Data$M2@i, j = Data$M2@j, x = Data$M2@x), "sfa/M2.csv")
write.csv(simdat$Loadings_pf, "sfa/Loadings.csv")
write.csv(simdat$Omega_sf, "sfa/Omega.csv")



## Define fixed effect parameters and their intitial values for
## optimization
##
## Truncated Cholesky decomp of the covariance matrix in vector
## form (rebuilt as matrix inside TMB).
Loadings_vec <- rep(1,Data$n_f*ncol(Y_sp)-Data$n_f*(Data$n_f-1)/2)
pars <- list(beta_p=log(apply(Y_sp,2, mean)),
             Loadings_vec=Loadings_vec,
             log_kappa=log(1),
             Omega_xf=matrix(0,nrow=mesh$n,ncol=Data$n_f))

## Compile and build model
t0 <- Sys.time()
compile("sfa/sfa.cpp")
dyn.load(dynlib('sfa/sfa'))
Obj <- MakeADFun(data=Data, parameters=pars, random='Omega_xf')
Obj$env$beSilent()                      # suppress console output
table(names(Obj$env$last.par))          # parameter dimensions
t1 <- Sys.time()

## Optimize model, intialize at par, fn is marginal likelihood
## function, and gr is gradient of marginal likelihood function
Opt <- nlminb(Obj$par, Obj$fn, Obj$gr,
              control=list(eval.max=10000, iter.max=10000, trace=10))
if(Opt$convergence !=0) warning('Possibly failed convergence')
t2 <- Sys.time()

## Do delta method calcs and get reported variables
(adrep <- sdreport(Obj))
as.data.frame(summary(adrep, 'fixed'))
report <- Obj$report()
t3 <- Sys.time()

## Benchmarks on a Dell Precision 7520 laptop
time.compile <- t1-t0
time.optimize <- t2-t1
time.sdcalcs <- t3-t2
c(time.compile, time.optimize, time.sdcalcs)
## > Time differences in secs
## [1] 44.04650 92.84046 23.08970


## ## Can run it through Stan trivially, but it's really slow and
## ## has some sign switching issues with the loadings parameters
## library(tmbstan)
## fit <- tmbstan(Obj, chains=3, open_progress=FALSE, cores=3,
##                iter=750, warmup=250,  control=list(max_treedepth=12))

## ## Plot data if desired
## library(ggplot2)
## library(dplyr)
## library(tidyr)
## data.frame(simdat$Loc_xy, Y_sp) %>%
##   pivot_longer(-(1:2), names_prefix='X', names_to='species',
##   values_to='abundance') %>%
## ggplot(aes(x,y, color=log(abundance))) + geom_point(alpha=.5) + facet_wrap('species')
