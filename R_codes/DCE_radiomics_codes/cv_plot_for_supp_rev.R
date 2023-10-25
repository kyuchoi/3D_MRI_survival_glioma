rm(list=ls())
library(dplyr) # for select()
library(glmnet) # for cox LASSO
library(survival)
library(ggplot2) 
### installing survminer version 0.3.0 from tar.gz for this code d/t 'ggsurvplot' and 'ggcoxadjustedcurves'
# remove.packages('survminer')
# install.packages('C:\\Users\\CNDLMembers\\Documents\\survminer-0.3.0.tar.gz', repos=NULL, type='source')
library(survminer)
library(ReporteRs)

set.seed(9873425) # marginal p-value result on test set: 9873425

# read my data
root_dir <- "D:\\Data\\SNUH\\¹Ú¿¤·¹³ª"
data_dir <- file.path(root_dir, 'data')
result_dir <- file.path(root_dir, 'result')

list_of_xdata <- c("x_ed_ktrans", "x_ed_Ve", "x_ed_Vp", "x_et_ktrans", "x_et_Ve", "x_et_Vp")

# setting
use.total <- TRUE # FALSE # 
num_subset <- 1 # 1-6 indicates each 851 features for each map: i.e. 1 indicates x_et_ktrans -> only 1,2,5 works
use.univariate.cox <- TRUE # FALSE # 
use.original.only <- TRUE # FALSE # 
use.scale <- TRUE # FALSE # 
p.thr <- 0.05
alpha <- 1 # elastic: 1 is LASSO (default), and 0 is Ridge: not good
split_ratio <- 0.7

if (use.total){
  num_subset <- 1 # 1-6 indicates each 851 features for each map: i.e. 1 indicates x_et_ktrans -> only 1,2,5 works
  x1 <- read.csv(file.path(data_dir, paste0(list_of_xdata[[num_subset]], '.csv')), row.names = NULL) # to remove index, put NULL not FALSE
  num_subset <- 2 # 1-6 indicates each 851 features for each map: i.e. 1 indicates x_et_ktrans -> only 1,2,5 works
  x2 <- read.csv(file.path(data_dir, paste0(list_of_xdata[[num_subset]], '.csv')), row.names = NULL) # to remove index, put NULL not FALSE
  num_subset <- 5 # 1-6 indicates each 851 features for each map: i.e. 1 indicates x_et_ktrans -> only 1,2,5 works
  x5 <- read.csv(file.path(data_dir, paste0(list_of_xdata[[num_subset]], '.csv')), row.names = NULL) # to remove index, put NULL not FALSE
  
  x <- cbind(x1,x2,x5)
} else {
  x <- read.csv(file.path(data_dir, paste0(list_of_xdata[[num_subset]], '.csv')), row.names = NULL)
}

x_clinical <- read.csv(file.path(data_dir, 'x_clinical.csv'))
IDH <- x_clinical[,3] # IDH for univariate t-test
time <- x_clinical[,5]
status <- x_clinical[,6]
status <- as.numeric(as.character(status))

if (use.scale){
  x<-scale(x)
}

raw.data <- data.frame(time, status, x)

# split train and valid set
split_num <- sort(sample(1:nrow(raw.data), size = round(nrow(raw.data) * split_ratio)))
x_train <- raw.data[split_num,]
x_test <- raw.data[-split_num,]

# save as seperate csv file for each train and test set
write.csv(x_train, file.path(data_dir, 'x_train.csv'), row.names = FALSE)
write.csv(x_test, file.path(data_dir, 'x_test.csv'), row.names = FALSE)

x.train <- read.csv(file.path(data_dir, 'x_train.csv'), row.names = NULL)

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
                           #return(exp(cbind(coef(x),confint(x))))
                         })
  res <- t(as.data.frame(univ_results, check.names = FALSE))
  as.data.frame(res)
}

# reduce 851 to 49 significant features using multiple univariate cox regression
res <- data.frame(res)
res$p.value <- as.numeric(as.character(res$p.value)) # ref: https://freshrimpsushi.tistory.com/497
res.subset <- subset(res, select = c(p.value), subset = (p.value<p.thr)) # ref: https://rfriend.tistory.com/49
sig.ft <- rownames(res.subset)
time <- x.train$time
status <- x.train$status
sig.data <-data.frame(x.train[sig.ft], time, status)

# get original/wavelet features only
if (use.original.only){
  sig.data <- select(sig.data, contains('original')) # wavelet: not good coef (only 1 0.14)
  sig.data <-data.frame(sig.data, time, status)
}

# cox lasso regression
x <- model.matrix(~.-time -status, sig.data)
fit <- glmnet(x, Surv(sig.data$time, sig.data$status), family='cox', alpha = alpha) # 1 is LASSO (default), and 0 is Ridge: not good
plot(fit, label=TRUE)
cv.fit <- cv.glmnet(x, Surv(sig.data$time, sig.data$status), family='cox', alpha = alpha)# maxit=1000)
plot(cv.fit)

cv.fit$lambda.min
cv.fit$lambda.1se

Coefficients <- coef(fit, s = cv.fit$lambda.min)
Active.Index <- which(Coefficients != 0)
Active.Coefficients <- Coefficients[Active.Index]

Active.Index
Active.Coefficients

active <- data.frame(Active.Index, Active.Coefficients)
abs.active <- abs(active)
abs.active.sort <- abs.active[order(-Active.Coefficients),]
abs.active.sort.select <- subset(abs.active.sort, 
                                 select = c(Active.Index, Active.Coefficients),
                                 subset = (Active.Coefficients>0.1))
select.ft.idx <- abs.active.sort.select$Active.Index
select.ft.coef <- abs.active.sort.select$Active.Coefficients
select.ft <- data.frame(sig.ft[select.ft.idx], select.ft.coef)

### get linear combination of selected features with corresponding coefficients to make radiomic score
select.ft.name <- as.character(select.ft$sig.ft.select.ft.idx) # at first it's factor, so always check out the class
select.ft.value <- x.train[select.ft.name]
score.vec <- t(apply(select.ft.value, 1, function(x) {x * select.ft.coef})) # a<-c(1,2,3); b<-c(3,2,1); a*b = (3 4 3)
score <- apply(score.vec, 1, sum)

# make group using binarized score with median score so that we can perform KM analysis with log-rank test
score.bin <- score
score.bin[score >= median(score)] <- 1
score.bin[score < median(score)] <- 0
x.train$group <- as.factor(score.bin)

# Do KM analysis for train set
fit <- survfit(Surv(time, status)~group, data=x.train)
gg.main.train <- ggsurvplot(fit, legend.title="radiomic score",legend.labs=c("low-risk","high-risk"),conf.int=T,pval=T,surv.median.line="hv",risk.table=T)
# for cumulative hazard
gg.cumhaz.train <- ggsurvplot(fit, legend.title="radiomic score",legend.labs=c("low-risk","high-risk"),fun="event")

gg.main.train
gg.cumhaz.train

# for log-rank test
log.rank.train <- survdiff(Surv(time,status)~group,data=x.train)
# to get p-value of log-rank test for train set 
p.log.rank.train <- pchisq(log.rank.train$chisq, df = 1, lower.tail = FALSE) # df: degrees of freedom, which is (k-1), when k= # of groups (here, 2)

###### Do KM analysis for test set
x.test <- read.csv(file.path(data_dir, 'x_test.csv'), row.names = NULL)

select.ft.value <- x.test[select.ft.name]
score.vec <- t(apply(select.ft.value, 1, function(x) {x * select.ft.coef})) # a<-c(1,2,3); b<-c(3,2,1); a*b = (3 4 3)
score <- apply(score.vec, 1, sum)

# make group using binarized score with median score so that we can perform KM analysis with log-rank test
score.bin <- score
score.bin[score >= median(score)] <- 1
score.bin[score < median(score)] <- 0
x.test$group <- score.bin # as.factor(score.bin) makes an error for drawing cox graphs 

# Do KM analysis for test set
fit <- survfit(Surv(time, status)~group, data=x.test)
gg.main.test <- ggsurvplot(fit, legend.title="radiomic score",legend.labs=c("low-risk","high-risk"),conf.int=T,pval=T,surv.median.line="hv",risk.table=T)
# for cumulative hazard
gg.cumhaz.test <- ggsurvplot(fit, legend.title="radiomic score",legend.labs=c("low-risk","high-risk"),fun="event")

gg.main.test
gg.cumhaz.test

# for log-rank test
log.rank.test <- survdiff(Surv(time,status)~group,data=x.test)
# to get p-value of log-rank test for test set 
p.log.rank.test <- pchisq(log.rank.test$chisq, df = 1, lower.tail = FALSE) # df: degrees of freedom, which is (k-1), when k= # of groups (here, 2)

# saving best split num as csv
write.csv(split_num, file.path(data_dir, 'best_split_num.csv'), row.names = FALSE)
# loaded.split.num <- read.csv(file.path(data_dir, 'best_split_num.csv'), col.names = 'split_num')

##### now load clinical variables
x_clinical <- read.csv(file.path(data_dir, 'x_clinical.csv'))
IDH <- x_clinical[,3] # IDH for univariate t-test
age <- x_clinical[,1]
sex <- x_clinical[,2]

x.train$IDH <- IDH[split_num]
x.test$IDH <- IDH[-split_num]

x.train$age <- age[split_num]
x.test$age <- age[-split_num]

x.train$sex <- sex[split_num]
x.test$sex <- sex[-split_num]

##### Do cox for test set
res.cox <- coxph(Surv(time, status) ~ group + IDH + age + sex, data = x.test) # instead of x.train
summary(res.cox)
ggforest(res.cox, data=x.test) # works only in survminer version 0.4.6 !!!
# plot(survfit(res.cox)) # simple plot not using ggplot2: but actually the same except color

### Plot the baseline survival function
ggsurvplot(survfit(res.cox), data = x.test, color = "#2E9FDF",
           ggtheme = theme_minimal()) # works only in survminer version 0.3.0 (which only can be installed via source tar.gz) !!!

### comparing cox curves between risk group: However, for p-value, better draw KM plots for each univariable
# ggsurvplot(survfit(res.cox), data = x.test, conf.int = TRUE, legend.labs=c("group=1", "group=2"),
#            ggtheme = theme_minimal()) # not working: don't know why, but can detour following codes

# Detour making dummy covariates: http://www.sthda.com/english/wiki/cox-proportional-hazards-model
x.test.group <- with(x.test,
                     data.frame(group = c(0, 1),
                                IDH = c(0, 0), 
                                age = rep(mean(age, na.rm = TRUE), 2),
                                sex = c(1, 1)
                     )
)
x.test.group
fit <- survfit(res.cox, newdata = x.test.group)

cox.surv <- ggsurvplot(fit, conf.int = TRUE) # FALSE
cox.surv

### OR you can just use old function called ggcoxadjustedcurves in survminer version 0.3.0
ggcoxadjustedcurves(res.cox, data = x.test,
                    individual.curves = TRUE)

# Adjusted survival curves for the variable "group"
ggcoxadjustedcurves(res.cox, data = x.test,
                    # conf.int = TRUE, # not working
                    variable  = x.test[, "group"],   # Variable of interest
                    legend.title = "group",        # Change legend title
                    palette = "npg",             # nature publishing group color palettes
                    curv.size = 2                # Change line size
)

# Adjusted survival curves for the variable "sex"
ggcoxadjustedcurves(res.cox, data = x.test,
                    variable  = x.test[, "sex"],   # Variable of interest
                    legend.title = "sex",        # Change legend title
                    palette = "npg",             # nature publishing group color palettes
                    curv.size = 2                # Change line size
)

# Adjusted survival curves for the variable "IDH"
ggcoxadjustedcurves(res.cox, data = x.test,
                    variable  = x.test[, "IDH"],   # Variable of interest
                    legend.title = "IDH",        # Change legend title
                    palette = "npg",             # nature publishing group color palettes
                    curv.size = 2                # Change line size
)

#### drawing to PPT: USE reporteR to preserve transparency of plot

# for train results
doc <- pptx()
doc <- addSlide(doc, "Two Content") # Title and Content # Two Content --> not making following error only in survminer version 0.4.6
##  Error in grid::grid.newpage() : pptx device only supports one page 
doc <- addPlot(doc, function() print(gg.main.train, newpage = FALSE), # MUST need , newpage = FALSE: or you get the above error
               vector.graphic = TRUE)
doc <- addPlot(doc, function() print(gg.cumhaz.train, newpage = FALSE),
               vector.graphic = TRUE)
writeDoc(doc, file = file.path(result_dir, "cox_risk_group_train.pptx"))

# for test results
doc <- pptx()
doc <- addSlide(doc, "Two Content") # Title and Content
doc <- addPlot(doc, function() print(gg.main.test, newpage = FALSE),
               vector.graphic = TRUE)
doc <- addPlot(doc, function() print(gg.cumhaz.test, newpage = FALSE),
               vector.graphic = TRUE)
writeDoc(doc, file = file.path(result_dir, "cox_risk_group_test.pptx"))

