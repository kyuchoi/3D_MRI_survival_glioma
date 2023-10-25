rm(list=ls())
library(dplyr) # for select()
library(glmnet) # for cox LASSO
library(survival)
library(ggplot2) 
library(rstudioapi) # for getActiveDocumentContext
### installing survminer version 0.3.0 from tar.gz for this code d/t 'ggsurvplot' and 'ggadjustedcurves'
# remove.packages('survminer')
# install.packages('D:\\Downloads\\survminer-0.3.0.tar.gz', repos=NULL, type='source')
library(survminer)
library(lmtest) # for lrtest() : https://api.rpubs.com/tomanderson_34/lrt

### read my data
current.path <- getActiveDocumentContext()
result_dir <- file.path(dirname(current.path$path), 'results')
data_dir <- file.path(dirname(current.path$path), 'data')
# root_dir <- dirname(dirname(dirname(current.path$path)))
# data_dir <- file.path(root_dir, 'data')

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

res.cox <- coxph(Surv(time, status) ~ group + IDH + MGMT + age + sex, data = x.test) # instead of x.train
summary(res.cox)

ggsurvplot(survfit(res.cox), data = x.test, conf.int = TRUE, legend.labs=c("group=1", "group=2"),
           ggtheme = theme_minimal()) # not working: don't know why, but can detour following codes