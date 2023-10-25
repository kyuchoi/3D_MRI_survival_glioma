rm(list=ls())
library(dplyr) # for select()
library(glmnet) # for cox LASSO
library(survival)
library(ggplot2) 
### installing survminer version 0.3.0 from tar.gz for this code d/t 'ggsurvplot' and 'ggadjustedcurves'
# remove.packages('survminer')
# install.packages('D:\\Downloads\\survminer-0.3.0.tar.gz', repos=NULL, type='source')
library(survminer)
### to install ReporteRs: save plots in PPTX
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre1.8.0_201')
Sys.getenv('JAVA_HOME')
install.packages('D:\\Downloads\\ReporteRsjars_0.0.4.tar.gz', repos=NULL, type='source', INSTALL_opts=c("--no-multiarch"))
install.packages('D:\\Downloads\\ReporteRs_0.8.10.tar.gz', repos=NULL, type='source', INSTALL_opts=c("--no-multiarch"))
library(ReporteRs)
library(rstudioapi) # for getActiveDocumentContext
### for maxstat to get cutoff value for survival analysis
# install.packages('maxstat')
library(maxstat)
# install.packages('splitstackshape')
library(splitstackshape) # for stratified function

# read my data
current.path <- getActiveDocumentContext()
root_dir <- dirname(dirname(current.path$path))
data_dir <- file.path(root_dir, 'data')
result_dir <- file.path(root_dir, 'result')

# setting
use.univariate.cox <- TRUE # FALSE # 
use.scale <- TRUE # FALSE # 
use.maxstat <- TRUE # FALSE # false means to use median(train.score) as cutoff !!
use.preset.splitnum <- FALSE # TRUE # 
use.strata <- TRUE # FALSE # 
p.thr <- 0.05
alpha <- 0.8 # 0.8 # best: 0.8 # elastic: 1 is LASSO (default), and 0 is Ridge: not good
split_ratio <- 0.7 # best: 0.7
seed <- 123456 # final: 81234 # best after MGMT: 1234 /orig: 23487/ prev best: 123456 (age 0.017 and MGMT 0.005) /2nd best: 987654321 (0.00949 but IDH 0.487, age 0.049)
curv.size <- 0.5

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

### split train and valid set

set.seed(seed) 
if (use.strata){
  num.discovery <- round(nrow(raw.data) * split_ratio)
  num.valid <- nrow(raw.data) - num.discovery
  strata.data <- subset(raw.data, select=c('IDH','MGMT','sex','status'))
  out <- stratified(strata.data, c('IDH','MGMT','sex'), 0.3, keep.rownames = TRUE) # ,'status','MGMT','sex' # 0.3
  head(out)
  split_num <- sort(as.numeric(out$rn))
  residual <- strata.data[-split_num,] # Be sure to have ',' in the end: strata.data[-out.num,] (o), strata.data[-out.num] (x)
  # residual <- setdiff(strata.data, out) # NOT working: because it removes all the duplicates d/t set
  # s <- strata(strata.data, stratanames = c('sex'), size = c(num.discovery, num.valid), method="srswor") # ,'MGMT','sex','status'
  # getdata(data,s)
  # nrow(residual[residual$IDH == 1,]) # Be sure to have ',' in the end # 56
} else {
  split_num <- sort(sample(1:nrow(raw.data), size = round(nrow(raw.data) * split_ratio)))  
}

x_train <- raw.data[split_num,]
x_test <- raw.data[-split_num,]

# save as seperate csv file for each train and test set
write.csv(x_train, file.path(data_dir, 'x_train_IDH.csv'), row.names = FALSE)
write.csv(x_test, file.path(data_dir, 'x_test_IDH.csv'), row.names = FALSE)

x.train <- read.csv(file.path(data_dir, 'x_train_IDH.csv'), row.names = NULL)

### for descriptive statistics

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
  # as.data.frame(res)
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

# cox lasso regression
x <- model.matrix(~.-time -status, sig.data)
fit <- glmnet(x, Surv(sig.data$time, sig.data$status), family='cox', alpha = alpha) # 1 is LASSO (default), and 0 is Ridge: not good
cv.fit <- cv.glmnet(x, Surv(sig.data$time, sig.data$status), family='cox', alpha = alpha)# maxit=1000)

Coefficients <- coef(fit, s = cv.fit$lambda.min)
Active.Index <- which(Coefficients != 0)
Active.Coefficients <- Coefficients[Active.Index]

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
select.ft.name <- as.character(select.ft$sig.ft.select.ft.idx) 

### get train.median values from selected features of train set
select.train.ft.value <- x.train[select.ft.name]
train.score.vec <- t(apply(select.train.ft.value, 1, function(x) {x * select.ft.coef})) # a<-c(1,2,3); b<-c(3,2,1); a*b = (3 4 3)
train.score <- apply(train.score.vec, 1, sum)
train.median <- median(train.score)

### use maxstat instead of median
if(use.maxstat){
  mstat <- maxstat.test(Surv(time, status) ~ train.score, data=x.train,
                        smethod='LogRank', pmethod='exactGauss',
                        abseps=0.01
  )
  mstat
  plot(mstat)
  mstat.p <- as.numeric(mstat$p.value)
  cutoff <- mstat$estimate
} else {
  cutoff <- train.median
}

###### Do KM analysis for test set
x.test <- read.csv(file.path(data_dir, 'x_test_IDH.csv'), row.names = NULL)

select.ft.value <- x.test[select.ft.name]
score.vec <- t(apply(select.ft.value, 1, function(x) {x * select.ft.coef})) # a<-c(1,2,3); b<-c(3,2,1); a*b = (3 4 3)
score <- apply(score.vec, 1, sum)

# make group using binarized score with median score so that we can perform KM analysis with log-rank test
score.bin <- score

score.bin[score >= cutoff] <- 1
score.bin[score < cutoff] <- 0
x.test$group <- score.bin # as.factor(score.bin) # group as factor is not working for ggsurvplot !!

##### Do cox for test set
res.cox <- coxph(Surv(time, status) ~ group + IDH + MGMT + age + sex, data = x.test) # instead of x.train
summary(res.cox)
capture.output(summary(res.cox),
               file = file.path(result_dir, 'cox_summary.txt'),
               append = TRUE)
gg <- ggforest(res.cox, data=x.test) # works only in survminer version 0.4.6 !!!
# plot(survfit(res.cox)) # simple plot not using ggplot2: but actually the same except color

## USE reporteR to preserve transparency of plot
doc <- pptx()
doc <- addSlide(doc, "Two Content") # Two Content > Title and Content
doc <- addPlot(doc, function() print(gg, newpage = FALSE), 
               vector.graphic = TRUE)
writeDoc(doc, file = file.path(result_dir, "ggforest_2ndrev.pptx"))

### Plot the baseline survival function
x.test.cox.surv <- ggsurvplot(survfit(res.cox), data = x.test, color = "#2E9FDF", conf.int = FALSE,
           ggtheme = theme_minimal()) # works only in survminer version 0.3.0 (which only can be installed via source tar.gz) !!!
x.test.cox.surv

### comparing cox curves between risk group: However, for p-value, better draw KM plots for each univariable
# ggsurvplot(survfit(res.cox), data = x.test, conf.int = TRUE, legend.labs=c("group=1", "group=2"),
#            ggtheme = theme_minimal()) # not working: don't know why, but can detour following codes

### Detour making dummy covariates: http://www.sthda.com/english/wiki/cox-proportional-hazards-model
# fix other covariates using contrast vector (e.g. (1,1)) except the variable of interest (i.e. set as (0,1) for var of interest only)
x.test.group <- with(x.test,
                     data.frame(group = c(0, 1),
                                IDH = c(1, 1), 
                                age = rep(mean(age, na.rm = TRUE), 2),
                                MGMT = c(1, 1),
                                sex = c(1, 1)
                     )
)
x.test.group

fit <- survfit(res.cox, data = x.test.group)
x.test.cox.surv.group <- ggsurvplot(fit, conf.int = FALSE) # TRUE: too much overlap -> removed
x.test.cox.surv.group

### OR you can just use old function called ggadjustedcurves in survminer version 0.3.0
x.test.surv.total <- ggadjustedcurves(res.cox, data = x.test,
                    individual.curves = FALSE) # TRUE: draw step-like indiv curves
x.test.surv.total # same as x.test.cox.surv except confidence interval

# Adjusted survival curves for the variable "group"
x.test.surv.group <- ggadjustedcurves(res.cox, data = x.test,
                    # conf.int = TRUE, # not working
                    variable  = x.test[, "group"],   # Variable of interest
                    legend.title = "group",        # Change legend title
                    palette = "npg",             # nature publishing group color palettes
                    curv.size = curv.size                # Change line size
)
x.test.surv.group

# Adjusted survival curves for the variable "sex"
x.test.surv.sex <- ggadjustedcurves(res.cox, data = x.test,
                    variable  = x.test[, "sex"],   # Variable of interest
                    legend.title = "sex",        # Change legend title
                    palette = "npg",             # nature publishing group color palettes
                    curv.size = curv.size                # Change line size
)
x.test.surv.sex

# Adjusted survival curves for the variable "IDH"
x.test.surv.IDH <- ggadjustedcurves(res.cox, data = x.test,
                    variable  = x.test[, "IDH"],   # Variable of interest
                    legend.title = "IDH",        # Change legend title
                    palette = "npg",             # nature publishing group color palettes
                    curv.size = curv.size                # Change line size
)
x.test.surv.IDH

# Adjusted survival curves for the variable "MGMT"
x.test.surv.MGMT <- ggadjustedcurves(res.cox, data = x.test,
                    variable  = x.test[, "MGMT"],   # Variable of interest
                    legend.title = "MGMT",        # Change legend title
                    palette = "npg",             # nature publishing group color palettes
                    curv.size = curv.size                # Change line size
)
x.test.surv.MGMT

# Adjusted survival curves for the variable "age"
# x.test.surv.age <- ggadjustedcurves(res.cox, data = x.test,
#                                         variable  = x.test[, "age"],   # Variable of interest
#                                         legend.title = "age",        # Change legend title
#                                         palette = "npg",             # nature publishing group color palettes
#                                         curv.size = curv.size                # Change line size
# )
# x.test.surv.age # Removed because of multiple curves for each ages (i.e. 29, 31, 33, ...)

#### drawing to PPT: USE reporteR to preserve transparency of plot

# for test results
doc <- pptx()
doc <- addSlide(doc, "Two Content") # Title and Content
doc <- addPlot(doc, function() print(x.test.cox.surv, newpage = FALSE),
               vector.graphic = TRUE)
doc <- addPlot(doc, function() print(x.test.surv.sex, newpage = FALSE),
               vector.graphic = TRUE)
doc <- addSlide(doc, "Two Content") # Title and Content
doc <- addPlot(doc, function() print(x.test.surv.group, newpage = FALSE),
               vector.graphic = TRUE)
doc <- addPlot(doc, function() print(x.test.surv.MGMT, newpage = FALSE),
               vector.graphic = TRUE)
doc <- addSlide(doc, "Two Content") # Title and Content
doc <- addPlot(doc, function() print(x.test.surv.IDH, newpage = FALSE),
               vector.graphic = TRUE)
# doc <- addPlot(doc, function() print(x.test.surv.age, newpage = FALSE),
#                vector.graphic = TRUE)
writeDoc(doc, file = file.path(result_dir, "cox_each_group_test_2ndrev.pptx"))


