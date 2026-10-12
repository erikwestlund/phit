# Binomial random-intercept marginal ML. R is a development reference only.
suppressPackageStartupMessages(library(lme4))
RNGkind('Mersenne-Twister','Inversion','Rejection');set.seed(20261011)
json<-function(x){
 if(is.list(x)){v<-vapply(x,json,'');if(!is.null(names(x)))return(paste0('{',paste(paste0('"',names(x),'":',v),collapse=','),'}'));return(paste0('[',paste(v,collapse=','),']'))}
 if(is.logical(x))return(if(x)'true' else 'false')
 if(is.character(x))return(paste0('"',gsub('"','\\"',x,fixed=TRUE),'"'))
 if(length(x)!=1)return(json(unname(as.list(x))))
 if(!is.finite(x))stop('nonfinite reference');sprintf('%.17g',x)
}
rows<-function(m)unname(lapply(seq_len(nrow(m)),function(i)as.list(unname(m[i,]))))
sizes<-c(12,18,8,22,15,11,20,9,17,13,24,16)
g<-rep(seq_along(sizes),sizes);n<-length(g)
study<-runif(n,1,12);practice<-.6*study+rnorm(n,sd=2)
x<-cbind(1,study,practice);b<-rnorm(length(sizes),sd=1.1)
eta<--2+.25*study+.08*practice+b[g];y<-rbinom(n,1,plogis(eta))
cases<-list(binary=list(x=x,y=y,groups=g,weights=rep(1,n),offset=rep(0,n)),
 no_intercept=list(x=x[,-1,drop=FALSE],y=y,groups=g,weights=rep(1,n),offset=rep(0,n)))
trials<-sample(5:30,n,TRUE);offset<-.3*sin(study)
cases$grouped_offset<-list(x=x,y=rbinom(n,trials,plogis(eta+offset))/trials,groups=g,weights=trials,offset=offset)
# Sparse events and small/unbalanced clusters, with genuine group heterogeneity.
g<-rep(1:16,times=rep(c(3,4,7,10),4));n<-length(g);predictor<-runif(n,-2,2)
y<-rbinom(n,1,plogis(-2.5+.6*predictor+rnorm(16,sd=1.8)[g]))
cases$rare_events<-list(x=cbind(1,predictor),y=y,groups=g,weights=rep(1,n),offset=rep(0,n))
g<-rep(1:14,each=8);n<-length(g);predictor<-runif(n,-1,1)
y<-rbinom(n,1,plogis(.2+.7*predictor+rnorm(14,sd=2.2)[g]))
cases$large_group_variation<-list(x=cbind(1,predictor),y=y,groups=g,weights=rep(1,n),offset=rep(0,n))
cases$boundary<-list(x=matrix(1,48,1),y=rep(c(0,1,0,1),12),groups=rep(1:12,each=4),weights=rep(1,48),offset=rep(0,48))
# Published lme4 example; period one is the baseline. Data already licensed via
# docs/data/lmm/COPYING (lme4 GPL >=2), with a separate provenance record.
d<-as.data.frame(cbpp);write.csv(d,'docs/data/cbpp.csv',row.names=FALSE)
cases$cbpp<-list(x=cbind(1,as.numeric(d$period==2),as.numeric(d$period==3),as.numeric(d$period==4)),
 y=d$incidence/d$size,groups=as.integer(d$herd),weights=d$size,offset=rep(0,nrow(d)))
fixtures<-list()
for(name in names(cases))for(points in c(1,9,25,61)){
 c<-cases[[name]];p<-ncol(c$x);d<-data.frame(y=c$y,weight=c$weights,offset=c$offset,group=factor(c$groups,levels=unique(c$groups)))
 for(j in seq_len(p))d[[paste0('x',j)]]<-c$x[,j]
 f<-as.formula(paste('y~0+',paste(paste0('x',1:p),collapse='+'),'+(1|group)'))
 warnings<-c()
 fit<-withCallingHandlers(glmer(f,data=d,family=binomial('logit'),weights=weight,offset=offset,nAGQ=points,
    control=glmerControl(optimizer='bobyqa',tolPwrss=1e-12,optCtrl=list(rhoend=1e-9,maxfun=100000))),
    warning=function(e){warnings<<-c(warnings,conditionMessage(e));invokeRestart('muffleWarning')},message=function(e)invokeRestart('muffleMessage'))
 beta<-fixef(fit);cv<-as.matrix(vcov(fit));se<-sqrt(diag(cv));z<-beta/se
 effects<-as.vector(ranef(fit)$group[as.character(unique(c$groups)),1])
 fixed<-as.vector(plogis(c$x%*%beta+c$offset));conditional<-as.vector(fitted(fit))
 saturated<-sum(dbinom(round(c$y*c$weights),c$weights,c$y,log=TRUE))
 # lme4 nAGQ>1 integrates exp(-deviance/2), dropping the saturated-model
 # likelihood constant. Restore it to compare normalized binomial likelihoods.
 correction<-if(points>1)saturated else 0
 ll<-as.numeric(logLik(fit))+correction;np<-p+1
 result<-list(coefficients=as.list(unname(beta)),covariance=rows(cv),std_errors=as.list(unname(se)),z_statistics=as.list(unname(z)),
   p_values=as.list(unname(2*pnorm(abs(z),lower.tail=FALSE))),conf_int=rows(cbind(beta-qnorm(.975)*se,beta+qnorm(.975)*se)),
   fixed_fitted=as.list(fixed),fitted=as.list(conditional),linear_predictor=as.list(as.vector(predict(fit,type='link'))),
   residuals=as.list(c$y-conditional),random_effects=as.list(effects),random_intercept_sd=as.numeric(getME(fit,'theta')),
   random_intercept_variance=as.numeric(getME(fit,'theta'))^2,log_likelihood=ll,neg2_log_likelihood=-2*ll,
   aic=2*np-2*ll,bic=log(length(c$y))*np-2*ll,n_parameters=np,df_residual=length(c$y)-np,singular=isSingular(fit))
 fixtures[[length(fixtures)+1]]<-list(name=paste0(name,'_',points),dataset=name,quadrature_points=points,x=rows(c$x),y=as.list(c$y),
    groups=as.list(c$groups),weights=as.list(c$weights),offset=as.list(c$offset),expected=result,
    reference=list(optimizer='bobyqa',tolPwrss=1e-12,rhoend=1e-9,optimizer_code=as.list(fit@optinfo$conv$opt),warnings=as.list(warnings),
        reported_log_likelihood=as.numeric(logLik(fit)),log_likelihood_correction=correction))
 cat(name,points,'\n',file=stderr())
}
writeLines(json(list(r=R.version.string,lme4=as.character(packageVersion('lme4')),cases=fixtures)),'tests/fixtures/logistic-mixed-r.json')
