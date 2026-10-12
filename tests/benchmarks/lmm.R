suppressPackageStartupMessages(library(lme4))
args<-commandArgs(trailingOnly=TRUE)
start<-proc.time()[['elapsed']];d<-read.csv(args[1]);load_seconds<-proc.time()[['elapsed']]-start
estimate<-function(){
order <- unique(d$group)
d$group <- factor(d$group,levels=order)
fixed <- paste(names(d)[-(1:2)],collapse='+')
fit <- lmer(as.formula(paste('y~0+',fixed,'+(1|group)')),data=d,REML=FALSE,
 control=lmerControl(optimizer='bobyqa',optCtrl=list(rhoend=1e-10,maxfun=100000)))
if(length(fit@optinfo$conv$lme4$messages) && !isSingular(fit,tol=1e-8)) stop(paste(fit@optinfo$conv$lme4$messages,collapse=';'))
beta <- fixef(fit); covariance <- as.matrix(vcov(fit)); se <- sqrt(diag(covariance)); z <- beta/se
x <- as.matrix(d[-(1:2)])
fixed_fitted <- as.vector(x%*%beta)
conditional <- as.vector(fitted(fit))
effects <- ranef(fit)$group[as.character(order),1]
variance <- sigma(fit)^2; tau <- as.numeric(VarCorr(fit)$group[1,1])
rows <- function(m) unname(lapply(seq_len(nrow(m)),function(i) unname(as.list(m[i,]))))
result <- list(coefficients=unname(as.list(beta)),covariance=rows(covariance),std_errors=unname(as.list(se)),
 z_statistics=unname(as.list(z)),p_values=unname(as.list(2*pnorm(abs(z),lower.tail=FALSE))),
 conf_int=rows(cbind(beta-qnorm(.975)*se,beta+qnorm(.975)*se)),
 fixed_fitted=as.list(fixed_fitted),fitted=as.list(conditional),residuals=as.list(d$y-conditional),
 marginal_residuals=as.list(d$y-fixed_fitted),random_effects=unname(as.list(effects)),
 residual_variance=variance,random_intercept_variance=tau,sigma=sqrt(variance),icc=tau/(tau+variance),
 log_likelihood=as.numeric(logLik(fit)),aic=AIC(fit),df_residual=nrow(d)-length(beta))
indices <- unique(floor(seq(0,nrow(d)-1,length.out=10)))+1
result$fitted_sample <- result$fitted[indices]; result$residuals_sample <- result$residuals[indices]
result$fixed_fitted_sample <- result$fixed_fitted[indices]; result$marginal_residuals_sample <- result$marginal_residuals[indices]
result$fitted <- result$residuals <- result$fixed_fitted <- result$marginal_residuals <- NULL
result
}
invisible(estimate());times<-numeric(as.integer(args[2]))
for(i in seq_along(times)){gc();start<-proc.time()[['elapsed']];result<-estimate();times[i]<-proc.time()[['elapsed']]-start}

json <- function(x) {
 if(is.list(x)) {values <- vapply(x,json,'');
  if(!is.null(names(x))) return(paste0('{',paste(paste0('"',names(x),'":',values),collapse=','),'}'))
  return(paste0('[',paste(values,collapse=','),']'))}
 if(is.character(x)) return(paste0('"',x,'"'))
 if(length(x)!=1) return(json(unname(as.list(x))))
 if(!is.finite(x)) stop('nonfinite reference'); sprintf('%.17g',x)
}
indices<-unique(floor(seq(0,nrow(d)-1,length.out=10)))
cat(json(list(version=paste(R.version.string,'lme4',as.character(packageVersion('lme4'))),BLAS=unname(extSoftVersion()['BLAS']),load_seconds=load_seconds,fit_seconds=as.list(times),sample_indices=as.list(indices),result=result)))
