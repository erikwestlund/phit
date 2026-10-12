# Snapshot selected published data columns; no generated observations.
options(digits=17)
root <- 'docs/data/lmm';dir.create(root,recursive=TRUE,showWarnings=FALSE)
export <- function(package,name,columns=NULL) {
 env<-new.env();data(list=name,package=package,envir=env);d<-get(name,envir=env)
 if(!is.null(columns))d<-d[columns]
 write.csv(d,file.path(root,paste0(name,'.csv')),row.names=FALSE,na='')
}
export('lme4','sleepstudy')
export('nlme','Orthodont')
export('nlme','Machines')
export('mlmRev','Hsb82',c('school','ses','meanses','sector','mAch'))
packages<-c('lme4','nlme','mlmRev')
cat(paste0('{',paste(sprintf('"%s":{"version":"%s","license":"%s"}',packages,
 vapply(packages,function(p)as.character(packageVersion(p)),''),
 vapply(packages,function(p)packageDescription(p)$License,'')),collapse=','),'}'))
