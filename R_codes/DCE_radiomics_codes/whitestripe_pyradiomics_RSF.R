# https://cran.r-project.org/web/packages/WhiteStripe/vignettes/Running_WhiteStripe.html
library(WhiteStripe)
library(oro.nifti)
library(randomForestSRC)
library(risksetROC)
library(caret)
library(neurobase)

t1 = file.path('/mnt/hdd/kschoi/GBM/SNUH_merged/resized_bhk/13392712/t1_bet.nii.gz')
img = readNIfTI(fname = t1, reorient = FALSE)
ws = whitestripe(img = img, type = "T1", stripped = TRUE)
norm = whitestripe_norm(img = img, indices = ws$whitestripe.ind)
writenii(norm, '/mnt/hdd/kschoi/GBM/SNUH_merged/resized_bhk/13392712/t1_ws_r.nii.gz')

#mask = ws$mask.img
#mask[mask == 0] = NA