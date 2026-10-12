# CSV columns y, group, x0, x1, ... . Marginal ML, one random intercept.
suppressPackageStartupMessages(library(lme4))
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1],na.strings='NA')
d <- d[complete.cases(d),,drop=FALSE]
order <- unique(d$group)
d$group <- factor(d$group,levels=order)
fixed <- paste(names(d)[-(1:2)],collapse='+')
backend <- if(length(args)>1)args[2] else 'lme4'
formula <- as.formula(paste('y~0+',fixed))
if(backend=='lme4') {
 fit <- lmer(as.formula(paste('y~0+',fixed,'+(1|group)')),data=d,REML=FALSE,
  control=lmerControl(optimizer='bobyqa',optCtrl=list(rhoend=1e-10,maxfun=100000)))
 if(length(fit@optinfo$conv$lme4$messages) && !isSingular(fit,tol=1e-8)) stop(paste(fit@optinfo$conv$lme4$messages,collapse=';'))
 beta <- fixef(fit); covariance <- as.matrix(vcov(fit))
 conditional <- as.vector(fitted(fit))
 effects <- ranef(fit)$group[as.character(order),1]
 variance <- sigma(fit)^2; tau <- as.numeric(VarCorr(fit)$group[1,1])
} else if(backend=='nlme') {
 fit <- nlme::lme(formula,random=~1|group,data=d,method='ML',
  control=nlme::lmeControl(tolerance=1e-10,msTol=1e-10,maxIter=1000,msMaxIter=1000,
                          opt='optim',optimMethod='BFGS'))
 beta <- nlme::fixed.effects(fit); covariance <- as.matrix(vcov(fit))
 conditional <- as.vector(fitted(fit,level=1))
 effects <- nlme::ranef(fit)[as.character(order),1]
 variance <- sigma(fit)^2
 # Extract the numeric covariance factor, not rounded VarCorr display strings.
 tau <- variance * nlme::pdMatrix(fit$modelStruct$reStruct[[1]])[1,1]
} else stop('unknown mixed-model reference backend')
se <- sqrt(diag(covariance)); z <- beta/se
x <- as.matrix(d[-(1:2)])
fixed_fitted <- as.vector(x%*%beta)
rows <- function(m) unname(lapply(seq_len(nrow(m)),function(i) unname(as.list(m[i,]))))
result <- list(coefficients=unname(as.list(beta)),covariance=rows(covariance),std_errors=unname(as.list(se)),
 z_statistics=unname(as.list(z)),p_values=unname(as.list(2*pnorm(abs(z),lower.tail=FALSE))),
 conf_int=rows(cbind(beta-qnorm(.975)*se,beta+qnorm(.975)*se)),
 fixed_fitted=as.list(fixed_fitted),fitted=as.list(conditional),residuals=as.list(d$y-conditional),
 marginal_residuals=as.list(d$y-fixed_fitted),random_effects=unname(as.list(effects)),
 residual_variance=variance,random_intercept_variance=tau,sigma=sqrt(variance),icc=tau/(tau+variance),
 log_likelihood=as.numeric(logLik(fit)),aic=AIC(fit),df_residual=nrow(d)-length(beta))
json <- function(x) {
 if(is.list(x)) {values <- vapply(x,json,'');
  if(!is.null(names(x))) return(paste0('{',paste(paste0('"',names(x),'":',values),collapse=','),'}'))
  return(paste0('[',paste(values,collapse=','),']'))}
 if(is.character(x)) return(paste0('"',x,'"'))
 if(length(x)!=1) return(json(unname(as.list(x))))
 if(!is.finite(x)) stop('nonfinite reference'); sprintf('%.17g',x)
}
cat(json(list(version=R.version.string,lme4=as.character(packageVersion('lme4')),
 backend=backend,nlme=if(backend=='nlme')as.character(packageVersion('nlme')) else 'unused',result=result)))
