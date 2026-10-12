# Run from the repository root. Save observations so readers need no RNG or R.
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(20261011)
n <- 8
area <- round(runif(n, min=35, max=110), 1)
bedrooms <- pmax(1, pmin(4, round((area - 15) / 25 + rnorm(n, 0, 0.4))))
noise <- rnorm(n, mean=0, sd=180)
rent <- round(650 + 18 * area + noise, 2)
observations <- data.frame(area_m2=area, bedrooms=bedrooms, monthly_rent=rent)
dir.create("docs/data", showWarnings=FALSE)
write.csv(observations, "docs/data/housing.csv", row.names=FALSE)
print(observations)
print(cor(observations))
