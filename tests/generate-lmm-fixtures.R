# Generate synthetic students clustered in schools, then save lme4 ML fits.
RNGkind('Mersenne-Twister','Inversion','Rejection');set.seed(20261015)
sizes <- c(18,24,15,22,20,27,19,21,16,25,23,20)
groups <- rep(seq_along(sizes),sizes); n <- length(groups)
hours <- round(runif(n,1,12),2)
prior <- round(pmin(95,pmax(20,40+3*hours+rnorm(n,0,8))),2)
b <- rnorm(length(sizes),0,6)
score <- round(50+1.8*hours+0.2*prior+b[groups]+rnorm(n,0,5),3)
write.csv(data.frame(school=sprintf('school-%02d',groups),study_hours=hours,prior_score=prior,score=score),
 'docs/data/schools.csv',row.names=FALSE)
cases <- list(simulated_schools=list(x=cbind(1,hours,prior),y=score,groups=groups*7+100),
 simple=list(x=cbind(1,hours),y=score,groups=groups),
 no_intercept=list(x=cbind(hours,prior),y=score,groups=groups),
 intercept_only=list(x=matrix(1,n,1),y=score,groups=groups),
 singleton_groups=list(x=cbind(1,hours,prior),y=score,groups=c(99,groups[-1])),
 boundary=list(x=matrix(1,12,1),y=rep(c(-1,0,1),4),groups=rep(1:4,each=3)))
json <- function(x) {
 if(is.list(x)){v<-vapply(x,json,'');if(!is.null(names(x)))return(paste0('{',paste(paste0('"',names(x),'":',v),collapse=','),'}'));return(paste0('[',paste(v,collapse=','),']'))}
 if(is.character(x)) return(paste0('"',x,'"'))
 if(length(x)!=1)return(json(unname(as.list(x))))
 sprintf('%.17g',x)
}
rows <- function(m) unname(lapply(seq_len(nrow(m)),function(i) unname(as.list(m[i,]))))
fixtures <- lapply(names(cases),function(name){
 c <- cases[[name]]
 d <- data.frame(y=c$y,group=c$groups,c$x);names(d)[-(1:2)]<-paste0('x',seq_len(ncol(c$x))-1)
 input<-tempfile(fileext='.csv');write.csv(d,input,row.names=FALSE)
 reference<-paste(system2('Rscript',c('docs/verifications/lmm.R',input),stdout=TRUE),collapse='')
 unlink(input)
 # Splice the runner's JSON object without adding jsonlite as a development dependency.
 paste0('{"name":"',name,'","x":',json(rows(c$x)),',"y":',json(as.list(c$y)),',"groups":',json(as.list(c$groups)),',"reference":',reference,'}')
})
writeLines(paste0('{"cases":[',paste(fixtures,collapse=','),']}'),'tests/fixtures/lmm-r.json')
