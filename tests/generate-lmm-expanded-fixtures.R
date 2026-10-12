# Checked-in fixtures need no R at test time. Four actual lme4 optimizer wrappers.
suppressPackageStartupMessages(library(lme4))
json <- function(x) {
 if(is.list(x)) {v<-vapply(x,json,'');if(!is.null(names(x)))return(paste0('{',paste(paste0('"',names(x),'":',v),collapse=','),'}'));return(paste0('[',paste(v,collapse=','),']'))}
 if(is.logical(x))return(if(x)'true' else 'false')
 if(is.character(x))return(paste0('"',gsub('"','\\"',x,fixed=TRUE),'"'))
 if(length(x)!=1)return(json(unname(as.list(x))))
 if(!is.finite(x))stop('nonfinite fixture');sprintf('%.17g',x)
}
rows<-function(m)unname(lapply(seq_len(nrow(m)),function(i)as.list(unname(m[i,]))))
term<-function(z,g,correlated=TRUE)list(z=as.matrix(z),groups=g,correlated=correlated)
set.seed(20261011)
# Balanced crossing with actual noisy correlated predictors and slope variation.
n<-144;g<-rep(1:12,each=12);h<-rep(rep(1:6,each=2),12)
a<-rnorm(n);b<-.6*a+rnorm(n,sd=.8)
x<-cbind(1,a,b);u<-rnorm(12,sd=1.2);v<-.3*u+rnorm(12,sd=.4);w<-rnorm(6,sd=.7)
y<-2+.8*a-.4*b+u[g]+v[g]*a+w[h]+rnorm(n,sd=.6)
base<-list(x=x,y=y,terms=list(term(cbind(1,a),g)),reml=FALSE,weights=rep(1,n),offset=rep(0,n))
cases<-list(random_slopes_ml=base,random_slopes_reml=base,independent_slopes=base,
 crossed=base,nested=base,weighted_offset=base,no_intercept=base,random_slope_only=base)
cases$random_slopes_reml$reml<-TRUE
cases$independent_slopes$terms[[1]]$correlated<-FALSE
cases$crossed$terms<-list(term(cbind(1,a),g),term(matrix(1,n,1),h))
cases$nested$terms<-list(term(matrix(1,n,1),g),term(matrix(1,n,1),g*10+h))
cases$weighted_offset$weights<-runif(n,.5,2)
cases$weighted_offset$offset<-.3*b
cases$weighted_offset$reml<-TRUE
cases$no_intercept$x<-cbind(a,b)
cases$random_slope_only$terms<-list(term(matrix(a,n,1),g))
d<-read.csv('docs/data/lmm/sleepstudy.csv')
for(reml in c(FALSE,TRUE))cases[[paste0('sleepstudy_',if(reml)'reml' else 'ml')]]<-list(
 x=cbind(1,d$Days),y=d$Reaction,terms=list(term(cbind(1,d$Days),as.integer(factor(d$Subject)))),
 reml=reml,weights=rep(1,nrow(d)),offset=rep(0,nrow(d)))
cases$boundary<-list(x=matrix(1,48,1),y=rep(c(-2,-1,1,2),12),
 terms=list(term(matrix(1,48,1),rep(1:12,each=4))),reml=TRUE,weights=rep(1,48),offset=rep(0,48))
# Repeated assessment scores nested in students/classes/schools/districts.
# Explicit composite IDs prevent local school/class/student IDs being merged.
district<-rep(1:8,each=32)
local_school<-rep(rep(1:2,each=16),8)
local_class<-rep(rep(1:2,each=8),16)
local_student<-rep(rep(1:2,each=4),32)
school<-district*100+local_school
classroom<-school*10+local_class
student<-classroom*10+local_student
nn<-length(student)
study<-runif(nn,1,12);practice<-.65*study+rnorm(nn,sd=2)
xx<-cbind(1,study,practice)
random_draw<-function(ids,sd)rnorm(length(unique(ids)),sd=sd)[match(ids,unique(ids))]
shared<-60+1.7*study+.3*practice+random_draw(district,5)+random_draw(school,3)+random_draw(classroom,2)
y3<-shared+rnorm(nn,sd=.8)
y4<-shared+random_draw(student,1.4)+rnorm(nn,sd=.8)
ones<-matrix(1,nn,1)
for(reml in c(FALSE,TRUE))for(levels in 3:4){
    groups<-list(district,school,classroom)
    if(levels==4)groups<-c(groups,list(student))
    cases[[paste0('nested_',levels,'_grouping_levels_',if(reml)'reml' else 'ml')]]<-list(
        x=xx,y=if(levels==3)y3 else y4,terms=lapply(groups,function(g)term(ones,g)),
        reml=reml,weights=rep(1,nn),offset=rep(0,nn))
}
optimizers<-c('bobyqa','Nelder_Mead','nlminbwrap','nloptwrap')
fixtures<-lapply(names(cases),function(name){
 c<-cases[[name]];n<-length(c$y);p<-ncol(c$x)
 d<-data.frame(y=c$y,weight=c$weights,offset=c$offset)
 for(j in seq_len(p))d[[paste0('x',j)]]<-c$x[,j]
 random<-c()
 for(t in seq_along(c$terms)){
  tm<-c$terms[[t]];gg<-paste0('g',t);d[[gg]]<-factor(tm$groups,levels=unique(tm$groups))
  zz<-paste0('z',t,'_',seq_len(ncol(tm$z)));d[zz]<-as.data.frame(tm$z)
  random<-c(random,paste0('(0+',paste(zz,collapse='+'),if(tm$correlated)'|' else '||',gg,')'))
 }
 f<-as.formula(paste('y~0+',paste(paste0('x',seq_len(p)),collapse='+'),'+',paste(random,collapse='+')))
 refs<-setNames(lapply(optimizers,function(optimizer){
  opt<-switch(optimizer,bobyqa=list(rhoend=1e-9,maxfun=100000),Nelder_Mead=list(FtolAbs=1e-10,XtolRel=1e-9,maxfun=100000),
    nlminbwrap=list(iter.max=10000,eval.max=100000,rel.tol=1e-9,x.tol=1e-8),
    nloptwrap=list(xtol_abs=1e-8,ftol_abs=0,ftol_rel=0,maxeval=100000))
  messages<-c()
  fit<-withCallingHandlers(lmer(f,data=d,REML=c$reml,weights=weight,offset=offset,
   control=lmerControl(optimizer=optimizer,optCtrl=opt,boundary.tol=1e-3)),warning=function(e){messages<<-c(messages,conditionMessage(e));invokeRestart('muffleWarning')},
   message=function(e)invokeRestart('muffleMessage'))
  restart_control<-NULL
  if(optimizer=='bobyqa' && grepl('_grouping_levels_',name)) {
    # A local restart improves variance-component precision on flat hierarchical
    # profiles. It uses only the original R fit, never native estimates.
    restart_control<-list(rhobeg=.001,rhoend=1e-9,maxfun=100000)
    fit<-withCallingHandlers(lmer(f,data=d,REML=c$reml,weights=weight,offset=offset,start=getME(fit,'theta'),
       control=lmerControl(optimizer=optimizer,optCtrl=restart_control,boundary.tol=1e-3)),
       warning=function(e){messages<<-c(messages,conditionMessage(e));invokeRestart('muffleWarning')},
       message=function(e)invokeRestart('muffleMessage'))
  }
  beta<-fixef(fit);cv<-as.matrix(vcov(fit));se<-sqrt(diag(cv));z<-beta/se
  vc<-VarCorr(fit);rr<-ranef(fit);terms<-lapply(seq_along(c$terms),function(t){
   # Independent columns are separate lme4 terms; assemble diagonal covariance.
   tm<-c$terms[[t]];k<-ncol(tm$z);gg<-paste0('g',t)
   indices<-which(sub('\\.[0-9]+$','',names(vc))==gg)
   cvt<-matrix(0,k,k)
   if(tm$correlated || k==1)cvt<-as.matrix(vc[[indices[1]]]) else {
     for(j in seq_len(k)){nm<-paste0('z',t,'_',j);ix<-indices[vapply(vc[indices],function(v)nm%in%rownames(v),FALSE)];cvt[j,j]<-vc[[ix]][1,1]}
   }
   ef<-as.matrix(rr[[gg]][as.character(unique(tm$groups)),paste0('z',t,'_',seq_len(k)),drop=FALSE])
   list(covariance=rows(cvt),effects=rows(ef))
  })
  list(result=list(coefficients=as.list(unname(beta)),covariance=rows(cv),std_errors=as.list(unname(se)),
   z_statistics=as.list(unname(z)),p_values=as.list(unname(2*pnorm(abs(z),lower.tail=FALSE))),
   conf_int=rows(cbind(beta-qnorm(.975)*se,beta+qnorm(.975)*se)),
   fixed_fitted=as.list(as.vector(c$x%*%beta+c$offset)),fitted=as.list(as.vector(fitted(fit))),
   residuals=as.list(as.vector(residuals(fit))),marginal_residuals=as.list(as.vector(c$y-c$x%*%beta-c$offset)),
   terms=terms,residual_variance=sigma(fit)^2,sigma=sigma(fit),log_likelihood=as.numeric(logLik(fit)),
   aic=AIC(fit),bic=BIC(fit),n_parameters=attr(logLik(fit),'df'),df_residual=df.residual(fit),singular=isSingular(fit)),
   optimizer=optimizer,control=opt,restart_control=if(is.null(restart_control))list() else restart_control,boundary_tolerance=1e-3,messages=as.list(messages),optimizer_code=as.list(fit@optinfo$conv$opt))
 }),optimizers)
 input<-list(name=name,x=rows(c$x),y=as.list(c$y),terms=lapply(c$terms,function(t)list(z=rows(t$z),groups=as.list(t$groups),correlated=t$correlated)),
 reml=c$reml,weights=as.list(c$weights),offset=as.list(c$offset),references=refs)
 cat(name,'\n',file=stderr());input
})
writeLines(json(list(r=R.version.string,lme4=as.character(packageVersion('lme4')),optimizers=as.list(optimizers),cases=fixtures)),
 'tests/fixtures/lmm-expanded-r.json')
