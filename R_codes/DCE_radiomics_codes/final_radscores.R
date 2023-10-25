rm(list=ls())
library(rstudioapi) # for getActiveDocumentContext
# read my data

current.path <- getActiveDocumentContext()
root_dir <- dirname(dirname(current.path$path))
data_dir <- file.path(root_dir, 'data')

# setting
x <- read.csv(file.path(data_dir, 'x_features.csv'), row.names = NULL)
x_scaled <- scale(x[,2:17])
# for (i in seq(1:16)){
#   x_i <- x_scaled[,i] # IDH for univariate t-test
# }
# status <- as.numeric(as.character(status))
# raw.data <- data.frame(time, status, IDH, MGMT, age, sex, x)

### coefficients
coef <- read.csv(file.path(data_dir, 'coef.csv'), row.names = NULL)

radscore1 <- sum(coef * x_scaled[10,])
radscore2 <- sum(coef * x_scaled[13,])
