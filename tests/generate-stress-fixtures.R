# Deterministic v0.1 edge cases. Reference is base R; fixtures need no R at runtime.
RNGkind('Mersenne-Twister','Inversion','Rejection'); set.seed(20261011)
rows<-function(m)unname(lapply(seq_len(nrow(m)),function(i)as.list(unname(m[i,]))))
json<-function(x){if(is.list(x)){v<-vapply(x,json,'');if(!is.null(names(x)))return(paste0('{',paste(paste0('"',names(x),'":',v),collapse=','),'}'));return(paste0('[',paste(v,collapse=','),']'))};if(is.logical(x))return(if(x)'true' else 'false');if(is.character(x))return(paste0('"',x,'"'));if(length(x)!=1)return(json(as.list(x)));if(!is.finite(x))stop('nonfinite fixture');sprintf('%.17g',x)}
a<-seq(-2,2,length.out=30); noise<-sin(seq_along(a))*0.2; base<-2+0.7*a+noise
cases<-list()
add<-function(name,model,x,y,weights=rep(1,length(y)),offset=rep(0,length(y)),absolute=1e-8,relative=1e-7){
 reference_warnings<-character()
 colnames(x)<-paste0('x',seq_len(ncol(x)))
 if(model=='ols'){
  fit<-lm(y~x-1,weights=weights,offset=offset);s<-summary(fit);stat<-s$coefficients[,3];pv<-s$coefficients[,4];ci<-confint(fit)
 }else{
  family<-switch(model,logistic=binomial('logit'),probit=binomial('probit'),poisson=poisson())
  start<-rep(0,ncol(x));j<-which(apply(x,2,function(v)all(v==1)))
  if(length(j))start[j[1]]<-family$linkfun(weighted.mean(y,weights))-mean(offset)
  reference_warnings<-character();fit<-withCallingHandlers(glm(y~x-1,weights=weights,offset=offset,family=family,start=start,control=glm.control(epsilon=1e-12,maxit=100)),warning=function(w){reference_warnings<<-c(reference_warnings,conditionMessage(w));invokeRestart('muffleWarning')});stopifnot(fit$converged)
  s<-summary(fit);stat<-s$coefficients[,3];pv<-s$coefficients[,4];ci<-confint.default(fit)
 }
 expected<-list(coefficients=as.list(unname(coef(fit))),covariance=rows(vcov(fit)),std_errors=as.list(unname(s$coefficients[,2])),
  statistics=as.list(unname(stat)),p_values=as.list(unname(pv)),conf_int=rows(ci),fitted=as.list(unname(fitted(fit))),
  residuals=as.list(unname(residuals(fit,type='response'))),log_likelihood=as.numeric(logLik(fit)))
 if(model!='ols'){expected$deviance<-deviance(fit);expected$aic<-AIC(fit)}
 cases[[length(cases)+1]]<<-list(name=name,model=model,x=rows(x),y=as.list(y),weights=as.list(weights),offset=as.list(offset),
  reference=list(implementation=if(model=='ols')'base R lm' else 'base R glm',family=model,formula='y ~ x - 1',
   warnings=as.list(reference_warnings),glm_epsilon=1e-12,glm_maxit=100,start='intercept initialized at marginal link minus mean offset',seed=20261011),
  tolerance=list(absolute=absolute,relative=relative),response_scale=if(model=='ols')max(abs(y)) else 1,expected=expected)
}
add('nearly_collinear','ols',cbind(1,a,a+0.001*cos(seq_along(a))),base,relative=1e-6)
add('poorly_scaled','ols',cbind(1,a*1e-120,cos(seq_along(a))*1e120),base)
add('small_outcome','ols',cbind(1,a),base*1e-140,relative=1e-6)
add('large_outcome','ols',cbind(1,a),base*1e140,relative=1e-6)
add('small_residual_variance','ols',cbind(1,a),2+0.7*a+noise*1e-6,absolute=1e-8,relative=1e-6)
add('weighted_no_intercept','ols',cbind(a,cos(seq_along(a))),base,weights=seq(0.1,3,length.out=30),offset=sin(a))
for(model in c('logistic','probit')){
 add(paste0(model,'_rare'),model,matrix(1,1000,1),c(1,rep(0,999)),relative=2e-6)
 for(o in c(-1000,1000))add(paste0(model,'_offset_',o),model,matrix(1,20,1),rep(c(0,1,0,0,1),4),offset=rep(o,20),absolute=1e-8,relative=2e-6)
}
for(model in c('logistic','probit'))add(paste0(model,'_extreme_known_predictors'),model,matrix(1,22,1),c(rep(c(0,1),10),0,1),offset=c(rep(0,20),-1000,1000),relative=1e-6)
add('poisson_tiny_known_means','poisson',matrix(c(1,0,0,0),4,1),c(2,0,0,0),offset=c(log(2),-50,-100,-500),relative=1e-6)
add('poisson_sparse','poisson',cbind(1,seq(-1,1,length.out=30)),as.numeric(seq_len(30)%in%c(7,21)),relative=1e-6)
add('poisson_large','poisson',matrix(1,6,1),rep(1e12,6),offset=rep(log(1e12),6),absolute=0.005,relative=1e-10)
for(o in c(-1000,1000))add(paste0('poisson_offset_',o),'poisson',matrix(1,20,1),rep(c(0,1,2,3,4),4),offset=rep(o,20),relative=1e-6)
writeLines(json(list(reference=R.version.string,generator='tests/generate-stress-fixtures.R',cases=cases)),'tests/fixtures/stress-r.json')
