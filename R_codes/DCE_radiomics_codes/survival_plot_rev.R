rm(list=ls())
library(dplyr) # for select()
library(glmnet) # for cox LASSO
library(survival)
library(ggplot2) 
### NEED 0.4.0 for survival_plot: no need to remove 0.3.0, just reboot R
# install.packages('survminer')
library(survminer) 
library(ReporteRs)
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("made4")
library(made4) # for heatplot
library(rstudioapi) # for getActiveDocumentContext

seed <- 123456
set.seed(seed) # 23487 # 57485 # orig: 987654321 # best: 654321 except coef > 1e+5 # best: 123456 except num of ft is only 2

# read my data
current.path <- getActiveDocumentContext()
root_dir <- dirname(dirname(current.path$path))
data_dir <- file.path(root_dir, 'data')
result_dir <- file.path(root_dir, 'result')

# setting
use.heatplot <- FALSE # TRUE # 
use.univariate.cox <- TRUE # FALSE # 
use.scale <- TRUE # FALSE # 
use.train.median <- TRUE # FALSE # 
p.thr <- 0.05
split_ratio <- 0.7
alpha <- 1
  
x <- read.csv(file.path(data_dir, 'x_total_preproc.csv'), row.names = NULL)

x_clinical <- read.csv(file.path(data_dir, 'x_clinical.csv'))
IDH <- x_clinical[,7] # IDH for univariate t-test
age <- x_clinical[,1]
sex <- x_clinical[,2]
MGMT <- x_clinical[,3]
time <- x_clinical[,5]
status <- x_clinical[,6]
status <- as.numeric(as.character(status))

if (use.scale){
  x<-scale(x)
}

raw.data <- data.frame(time, status, IDH, MGMT, age, sex, x)

# split train and valid set
split_num <- sort(sample(1:nrow(raw.data), size = round(nrow(raw.data) * split_ratio)))
# split_num <- split_num <- c(2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 16, 17, 19, 20, 21, 22, 25, 26, 27, 28, 30, 32, 35, 36, 38, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 60, 61, 62, 63, 65, 66, 67, 69, 70, 71, 72, 73, 79, 80, 83, 84, 86, 87, 88, 90, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 115, 116, 117, 118, 119, 121, 122, 123, 126, 130, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142, 143, 144, 149)
# split_num <- sort(split_num)

x_train <- raw.data[split_num,]
x_test <- raw.data[-split_num,]

# save as seperate csv file for each train and test set
write.csv(x_train, file.path(data_dir, 'x_train_scaled_survplot.csv'), row.names = FALSE)
write.csv(x_test, file.path(data_dir, 'x_test_scaled_survplot.csv'), row.names = FALSE)

x.train <- read.csv(file.path(data_dir, 'x_train_scaled_survplot.csv'), row.names = NULL)

if (use.univariate.cox){
  covariates <- colnames(x.train) 
  univ_formulas <- sapply(covariates,
                          function(x) as.formula(paste('Surv(time, status)~', x)))
  
  univ_models <- lapply( univ_formulas, function(x){coxph(x, data = x.train)})
  # Extract data 
  univ_results <- lapply(univ_models,
                         function(x){ 
                           x <- summary(x)
                           p.value<-signif(x$wald["pvalue"], digits=2)
                           wald.test<-signif(x$wald["test"], digits=2)
                           beta<-signif(x$coef[1], digits=2);#coeficient beta
                           HR <-signif(x$coef[2], digits=2);#exp(beta)
                           HR.confint.lower <- signif(x$conf.int[,"lower .95"], 2)
                           HR.confint.upper <- signif(x$conf.int[,"upper .95"],2)
                           HR <- paste0(HR, " (", 
                                        HR.confint.lower, "-", HR.confint.upper, ")")
                           res<-c(beta, HR, wald.test, p.value)
                           names(res)<-c("beta", "HR (95% CI for HR)", "wald.test", 
                                         "p.value")
                           return(res)
                         })
  res <- t(as.data.frame(univ_results, check.names = FALSE))
  as.data.frame(res)
}

# reduce 851 to 49 significant features using multiple univariate cox regression
res <- data.frame(res)
res$p.value <- as.numeric(as.character(res$p.value)) 
res.subset <- subset(res, select = c(p.value), subset = (p.value<p.thr)) 
sig.ft <- rownames(res.subset)
time <- x.train$time
status <- x.train$status
sig.data <-data.frame(x.train[sig.ft], time, status)

# get original features only
sig.data <- select(sig.data, contains('original')) # wavelet: not good 
sig.data <-data.frame(sig.data, time, status)

# get heatmap of univariate selected 63 (original) features
if (use.heatplot){
  heatplot(scale(subset(sig.data, select=-c(time,status))))
}

# cox lasso regression
x <- model.matrix(~.-time -status, sig.data)
fit <- glmnet(x, Surv(sig.data$time, sig.data$status), family='cox', alpha = alpha) 
plot(fit, label=TRUE)
cv.fit <- cv.glmnet(x, Surv(sig.data$time, sig.data$status), family='cox', alpha = alpha)
plot(cv.fit)

Coefficients <- coef(fit, s = cv.fit$lambda.min)
Active.Index <- which(Coefficients != 0)
Active.Coefficients <- Coefficients[Active.Index]

# to get active features including final 5 features
active <- data.frame(Active.Index, Active.Coefficients)
abs.active <- abs(active)
abs.active.sort <- abs.active[order(-Active.Coefficients),]
active.ft.coef <- abs.active.sort$Active.Coefficients
active.ft.idx <- abs.active.sort$Active.Index
active.ft <- data.frame(sig.ft[active.ft.idx], active.ft.coef)
active.ft.name <- as.character(active.ft$sig.ft.active.ft.idx)
df.active.ft <- data.frame(active.ft.name, active.ft.coef)

#### saving active features including final 5 features
write.csv(df.active.ft, file.path(result_dir, 'active_features.csv'))

abs.active.sort.select <- subset(abs.active.sort, 
                                 select = c(Active.Index, Active.Coefficients),
                                 subset = (Active.Coefficients>0.1 & Active.Coefficients<10.0)) # MUST: & Active.Coefficients<10.0
select.ft.idx <- abs.active.sort.select$Active.Index

# to get 5 feature names only
select.ft.coef <- abs.active.sort.select$Active.Coefficients
select.ft <- data.frame(sig.ft[select.ft.idx], select.ft.coef)

### get linear combination of selected features with corresponding coefficients to make radiomic score
select.ft.name <- as.character(select.ft$sig.ft.select.ft.idx)
select.ft.value <- x.train[select.ft.name]
score.vec <- t(apply(select.ft.value, 1, function(x) {x * select.ft.coef})) 
score <- apply(score.vec, 1, sum)

# make group using binarized score with median score so that we can perform KM analysis with log-rank test
score.bin <- score
score.bin[score >= median(score)] <- 1
score.bin[score < median(score)] <- 0
train.median <- median(score)
x.train$group <- as.factor(score.bin)

# Do KM analysis for train set
fit <- survfit(Surv(time, status)~group, data=x.train)
gg.main.train <- ggsurvplot(fit, legend.title="radiomic score",legend.labs=c("low-risk","high-risk"),conf.int=T,pval=T,surv.median.line="hv",risk.table=T) # risk.table="nrisk_cumevents"
# for cumulative hazard
gg.cumhaz.train <- ggsurvplot(fit, legend.title="radiomic score",legend.labs=c("low-risk","high-risk"),fun="event")

gg.main.train
gg.cumhaz.train

# for log-rank test
log.rank.train <- survdiff(Surv(time,status)~group,data=x.train)
# to get p-value of log-rank test for train set 
p.log.rank.train <- pchisq(log.rank.train$chisq, df = 1, lower.tail = FALSE) # df: degrees of freedom, which is (k-1), when k= # of groups (here, 2)

###### Do KM analysis for test set
x.test <- read.csv(file.path(data_dir, 'x_test_scaled_survplot.csv'), row.names = NULL)
# x.test <- scale(x.test)
select.ft.value <- x.test[select.ft.name]
score.vec <- t(apply(select.ft.value, 1, function(x) {x * select.ft.coef})) 
score <- apply(score.vec, 1, sum)

# make group using binarized score with median score so that we can perform KM analysis with log-rank test
score.bin <- score

if(use.train.median){
  score.bin[score >= train.median] <- 1
  score.bin[score < train.median] <- 0
  x.test$group <- score.bin # as.factor(score.bin) makes an error for drawing cox graphs 
} else {
  score.bin[score >= median(score)] <- 1
  score.bin[score < median(score)] <- 0
  x.test$group <- score.bin # as.factor(score.bin) makes an error for drawing cox graphs 
}

# Do KM analysis for test set
fit <- survfit(Surv(time, status)~group, data=x.test)
gg.main.test <- ggsurvplot(fit, legend.title="radiomic score",legend.labs=c("low-risk","high-risk"),conf.int=T,pval=T,surv.median.line="hv",risk.table=T) # risk.table="nrisk_cumevents"
# for cumulative hazard
gg.cumhaz.test <- ggsurvplot(fit, legend.title="radiomic score",legend.labs=c("low-risk","high-risk"),fun="event")

gg.main.test
gg.cumhaz.test

# for log-rank test
log.rank.test <- survdiff(Surv(time,status)~group,data=x.test)
# to get p-value of log-rank test for test set 
p.log.rank.test <- pchisq(log.rank.test$chisq, df = 1, lower.tail = FALSE) # df: degrees of freedom, which is (k-1), when k= # of groups (here, 2)

#### drawing to PPT: USE reporteR to preserve transparency of plot

# for train results
doc <- pptx()
doc <- addSlide(doc, "Two Content") 
doc <- addPlot(doc, function() print(gg.main.train, newpage = FALSE), 
               vector.graphic = TRUE)
doc <- addPlot(doc, function() print(gg.cumhaz.train, newpage = FALSE),
               vector.graphic = TRUE)
writeDoc(doc, file = file.path(result_dir, "cox_risk_group_train.pptx"))

# for test results
doc <- pptx()
doc <- addSlide(doc, "Two Content")
doc <- addPlot(doc, function() print(gg.main.test, newpage = FALSE),
               vector.graphic = TRUE)
doc <- addPlot(doc, function() print(gg.cumhaz.test, newpage = FALSE),
               vector.graphic = TRUE)
writeDoc(doc, file = file.path(result_dir, "cox_risk_group_test.pptx"))

#### saving selected 5 features
write.csv(select.ft, file.path(result_dir, 'selected_features.csv'))
