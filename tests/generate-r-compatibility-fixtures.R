# Explicit designs, prior weights and offsets for every current fixed estimator.
json<-function(x){if(is.list(x)){v<-vapply(x,json,'');if(!is.null(names(x)))return(paste0('{',paste(paste0('"',names(x),'":',v),collapse=','),'}'));return(paste0('[',paste(v,collapse=','),']'))};if(is.character(x))return(paste0('"',x,'"'));if(length(x)!=1)return(json(unname(as.list(x))));if(!is.finite(x))stop('nonfinite fixture');sprintf('%.17g',x)}
rows<-function(m)unname(lapply(seq_len(nrow(m)),function(i)as.list(unname(m[i,]))))
set.seed(20261011);n<-180
study<-runif(n,1,12);practice<-.55*study+rnorm(n,sd=2)
x<-cbind(1,study,practice);offset<-.15*sin(study)
cases<-list()
for(model in c('ols','logistic','probit','poisson'))for(intercept in c(TRUE,FALSE))for(grouped in if(model%in%c('logistic','probit'))c(FALSE,TRUE) else FALSE) {
 eta<--1+.22*study+.09*practice+offset
 y<-switch(model,ols=20+1.4*study+.5*practice+offset+rnorm(n,sd=2),logistic=rbinom(n,1,plogis(eta)),probit=rbinom(n,1,pnorm(eta)),poisson=rpois(n,exp(eta)))
 weights<-if(model%in%c('logistic','probit'))sample(1:5,n,TRUE) else runif(n,.3,3)
 if(grouped){weights<-sample(5:25,n,TRUE);y<-rbinom(n,weights,if(model=='logistic')plogis(eta) else pnorm(eta))/weights}
 xx<-if(intercept)x else x[,-1,drop=FALSE]
 d<-data.frame(y=y,weight=weights,offset=offset)
 for(j in seq_len(ncol(xx)))d[[paste0('x',j)]]<-xx[,j]
 # Use actual formula intercept so R summary follows centered/null conventions.
 fixed<-paste(paste0('x',if(intercept)2:ncol(xx) else 1:ncol(xx)),collapse='+')
 formula<-as.formula(paste('y~',if(!intercept)'0+' else '',fixed))
 fit<-if(model=='ols')lm(formula,data=d,weights=weight,offset=offset) else glm(formula,data=d,weights=weight,offset=offset,
  family=switch(model,logistic=binomial('logit'),probit=binomial('probit'),poisson=poisson('log')),control=glm.control(epsilon=1e-12,maxit=200))
 sm<-summary(fit);co<-sm$coefficients;cv<-as.matrix(vcov(fit));beta<-coef(fit)
 ref<-list(coefficients=as.list(unname(beta)),covariance=rows(cv),std_errors=as.list(unname(co[,2])),
  p_values=as.list(unname(co[,4])),fitted=as.list(as.vector(fitted(fit))),residuals=as.list(as.vector(residuals(fit,type='response'))),
  log_likelihood=as.numeric(logLik(fit)),df_residual=df.residual(fit))
 if(model=='ols')ref<-c(ref,list(t_statistics=as.list(unname(co[,3])),conf_int=rows(confint(fit)),rss=sum(weights*residuals(fit)^2),residual_variance=sm$sigma^2,sigma=sm$sigma,r_squared=sm$r.squared,adj_r_squared=sm$adj.r.squared))
 else ref<-c(ref,list(z_statistics=as.list(unname(co[,3])),conf_int=rows(cbind(beta-qnorm(.975)*co[,2],beta+qnorm(.975)*co[,2])),deviance=fit$deviance,null_deviance=fit$null.deviance,aic=AIC(fit)))
 cases[[length(cases)+1]]<-list(name=paste(model,if(intercept)'intercept' else 'no_intercept',if(grouped)'grouped' else 'individual',sep='_'),model=model,
 x=rows(xx),y=as.list(y),weights=as.list(weights),offset=as.list(offset),expected=ref)
}
writeLines(json(list(r=R.version.string,cases=cases)),'tests/fixtures/r-compatibility.json')
