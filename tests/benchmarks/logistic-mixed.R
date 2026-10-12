suppressPackageStartupMessages(library(lme4))
args<-commandArgs(trailingOnly=TRUE);start<-proc.time()[['elapsed']];d<-read.csv(args[1]);load_seconds<-proc.time()[['elapsed']]-start
d$group<-factor(d$group,levels=unique(d$group));indices<-unique(floor(seq(0,nrow(d)-1,length.out=10)))+1
rows<-function(m)unname(lapply(seq_len(nrow(m)),function(i)as.list(unname(m[i,]))))
estimate<-function(){
 first<-glmer(y~0+x0+x1+x2+(1|group),d,family=binomial(),nAGQ=1,
    control=glmerControl(optimizer='bobyqa',calc.derivs=FALSE,tolPwrss=1e-12,optCtrl=list(rhoend=1e-9,maxfun=100000)))
 # Explicit R-only local restart; force the joint Hessian even at large n.
 fit<-glmer(y~0+x0+x1+x2+(1|group),d,family=binomial(),nAGQ=1,
    start=list(theta=getME(first,'theta'),fixef=fixef(first)),
    control=glmerControl(optimizer='bobyqa',nAGQ0initStep=FALSE,calc.derivs=TRUE,tolPwrss=1e-12,
      optCtrl=list(rhobeg=.001,rhoend=1e-9,maxfun=100000)))
 if(length(fit@optinfo$conv$lme4$messages))stop(paste(fit@optinfo$conv$lme4$messages,collapse=';'))
 beta<-fixef(fit);cv<-as.matrix(vcov(fit));se<-sqrt(diag(cv));z<-beta/se
 conditional<-as.vector(fitted(fit));linear<-as.vector(predict(fit,type='link'));fixed<-as.vector(plogis(as.matrix(d[c('x0','x1','x2')])%*%beta))
 result<-list(coefficients=as.list(unname(beta)),covariance=rows(cv),std_errors=as.list(unname(se)),
   random_intercept_sd=as.numeric(getME(fit,'theta')),log_likelihood=as.numeric(logLik(fit)),
   fitted_sample=as.list(conditional[indices]),residuals_sample=as.list((d$y-conditional)[indices]),
   fixed_fitted_sample=as.list(fixed[indices]),linear_predictor_sample=as.list(linear[indices]),
   random_effects=as.list(as.vector(ranef(fit)$group[,1])),converged=TRUE)
 result
}
invisible(estimate());times<-numeric(as.integer(args[2]))
for(i in seq_along(times)){gc();start<-proc.time()[['elapsed']];result<-estimate();times[i]<-proc.time()[['elapsed']]-start}
json<-function(x){if(is.list(x)){v<-vapply(x,json,'');if(!is.null(names(x)))return(paste0('{',paste(paste0('"',names(x),'":',v),collapse=','),'}'));return(paste0('[',paste(v,collapse=','),']'))};if(is.logical(x))return(if(x)'true' else 'false');if(is.character(x))return(paste0('"',x,'"'));if(length(x)!=1)return(json(as.list(x)));if(!is.finite(x))stop('nonfinite benchmark');sprintf('%.17g',x)}
cat(json(list(version=paste(R.version.string,'lme4',packageVersion('lme4')),load_seconds=load_seconds,fit_seconds=as.list(times),sample_indices=as.list(indices-1),result=result)))
