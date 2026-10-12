args <- commandArgs(trailingOnly=TRUE)
start <- proc.time()[['elapsed']]
input <- jsonlite::fromJSON(args[1])
x <- input$draws
w <- input$weights / sum(input$weights)
source <- input$source_covariance
target <- input$target_covariance
a <- input$matrix
b <- input$shift
load <- proc.time()[['elapsed']] - start
moments <- function(unbiased=FALSE) {
  m <- colSums(x * w)
  centered <- sweep(x, 2, m)
  covariance <- crossprod(centered * sqrt(w))
  if (unbiased) covariance <- covariance / (1 - sum(w*w))
  list(mean=unname(m), covariance=unname(covariance))
}
covariance_map <- function() {
  ls <- t(chol(source)); lt <- t(chol(target))
  map <- t(backsolve(t(ls), t(lt)))
  list(matrix=unname(map), log_abs_det=sum(log(diag(lt)) - log(diag(ls))))
}
affine <- function() {
  list(draws=unname(sweep(x %*% t(a), 2, b, '+')),
       log_abs_det=as.numeric(determinant(a, logarithm=TRUE)$modulus))
}
calls <- list(population=function() moments(), unbiased=function() moments(TRUE),
              covariance_map=covariance_map, affine=affine)
if (args[2] == 'verify') {
  results <- lapply(calls, function(f) f())
  cat(jsonlite::toJSON(results, auto_unbox=TRUE, digits=NA, matrix='rowmajor'))
} else {
  times <- lapply(names(calls), function(key) {
    f <- calls[[key]]
    invisible(f())
    replicate(as.integer(args[3]), {
      start <- proc.time()[['elapsed']]
      for (j in seq_len(input$batches[[key]])) result <- f()
      (proc.time()[['elapsed']] - start) / input$batches[[key]]
    }, simplify=FALSE)
  })
  names(times) <- names(calls)
  cat(jsonlite::toJSON(list(version=R.version.string, jsonlite=as.character(packageVersion('jsonlite')),
    blas=extSoftVersion()[['BLAS']], load_seconds=load, seconds=times), auto_unbox=TRUE, digits=NA))
}
