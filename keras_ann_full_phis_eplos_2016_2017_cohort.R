

set.seed(57006)

rm(list=ls())

library(plyr)
library(dplyr)

library(keras)
#need to set eed for CPU/GPU random variation in processing as well as set learning rate very low to limti variation in learning (stochasticity within optimizer)
use_session_with_seed(57006) #https://github.com/rstudio/tensorflow/blob/master/R/seed.R 

library(caret)
library(tfestimators)
library(tfruns)
library(lubridate)
library(pROC)
library(ggplot2)
library(gridExtra)
library(reshape2)
#library(extrafont)

#guidance on font embedding (Note that if requiring Arial, default Helvetica is virtually the same in ggplot2)
#https://github.com/wch/extrafont 
# loadfonts(device="win")
# Sys.setenv(R_GSCMD = "C:/Program Files/gs/gs9.23/bin/gswin64c.exe")
# embed_fonts("font_ggplot.pdf", outfile="font_ggplot_embed.pdf")

setwd("C:/Users/Owner/Documents/MCHS/ELOS/PHIS/")

###################################################
#load data
###################################################

#load("~/MCHS/ELOS/PHIS/keras_ann_full_phis_eplos_2016_2017_cohortV1.RData")


tempall = read.csv("phis_eplos_2016_2017_cohort.csv")[,-1]
temp = tempall[tempall$age_yrs<=.08333333,]
rm(tempall)
table(temp$Complex_Chronic_Condition_Flag)

glimpse(temp)
summary(temp[,1:30])

str(temp$Discharge_Date)
temp$Discharge_Date = as.character(temp$Discharge_Date)
temp$Discharge_Date = ymd(temp$Discharge_Date)


#########################################################
#########################################################
#########################################################
#FUNCTIONS
#########################################################
#########################################################
#########################################################

#########################################################
#min max normalization function
#########################################################

min_max_normalization = function(x) {
  normalized = (x-min(x, na.rm=TRUE))/(max(x, na.rm=TRUE)-min(x, na.rm=TRUE))
  return(normalized)
}

equal_category_wt_construction = function(VARIABLE_TO_WEIGHT){
  tabs = data.frame(table(VARIABLE_TO_WEIGHT, useNA = "ifany"))
  tabs$n_mean = mean(tabs$Freq, na.rm=TRUE)
  tabs$p_mean = tabs$Freq/tabs$n_mean
  tabs$wt_total = 1/tabs$p_mean
  varwtdat = subset(merge(data.frame(VARIABLE_TO_WEIGHT = VARIABLE_TO_WEIGHT), 
                          tabs, by=c("VARIABLE_TO_WEIGHT"), all.x=TRUE),
                    select=c(wt_total))
  retlist = list(weight_summary = tabs, wt = varwtdat)
  return(retlist)
}

###############################################################
#miscellaneous cleanup
###############################################################

temp$LOSmonths = NA
temp$LOSmonths[temp$LOS>=0 & temp$LOS<30] = "a. < 1 month"
temp$LOSmonths[temp$LOS>=30 & temp$LOS<60] = "b. 1 to 2 months"
temp$LOSmonths[temp$LOS>=60 & temp$LOS<90] = "c. 2 to 3 months"
temp$LOSmonths[temp$LOS>=90 & temp$LOS<120] = "d. 3 to 4 months"
temp$LOSmonths[temp$LOS>=120 & temp$LOS<150] = "e. 4 to 5 months"
temp$LOSmonths[temp$LOS>=150 & temp$LOS<180] = "f. 5 to 6 months"
#temp$LOSmonths[temp$LOS>=30 & temp$LOS<180] = "b. 1 to 6 months"
temp$LOSmonths[temp$LOS>=180] = "g. greater than 6 months"
temp$LOSmonths = factor(temp$LOSmonths)
data.frame(table(temp$LOSmonths))

###############################################################
#develop indicators of stratification for train/validation/test
###############################################################

dput(names(temp))

variables_to_include = c("LOSmonths", 
                         #"Patient_Type_Title", 
                         #"Gender_Title", "Ethnicity_Title", "Race.White", "Race.Black", 
                         #"Race.Asian", "Race.Pacific_Islander", "Race.American_Indian", "Race.Other", 
                         #"admit_type", "insurance", "age_yrs", 
                         grep("DX_", names(temp), value=TRUE),
                         grep("PX_", names(temp), value=TRUE))
variables_to_include[1:20]


##################################################
#create test and train datsets that are temporal
##################################################

#test data are chronologically later (discharges from the last 6 months of the rolling-cohort)
testdat = temp[temp$Discharge_Date>="2017-06-01", which(names(temp) %in% variables_to_include)]
data.frame(table(testdat$LOSmonths))
#training data aret he first 18 monhts of discharges between 1/1/2016 and 6/1/2017
traindat = temp[temp$Discharge_Date<"2017-06-01", which(names(temp) %in% variables_to_include)]
data.frame(table(traindat$LOSmonths))

names(traindat[,1:30])
glimpse(traindat[,1:15])


#create weight from outcome to ensure equal costs among all multinomial categories
weightests = equal_category_wt_construction(traindat$LOSmonths)
weightests$weight_summary
head(weightests$wt)

#check to make sure that weights constructed above for cost function are same length as input data for modeling
nrow(traindat)
nrow(weightests$wt)

##########################################################
#SCALE CONTINUOUS VARIABLES BEFORE K-Fold Cross Validation
##########################################################

# age_yrs_descriptives = psych::describe(traindat$age_yrs, na.rm=TRUE)
# age_yrs_descriptives
# 
# traindat$age_yrs = min_max_normalization(traindat$age_yrs)
# psych::describe(traindat$age_yrs)

##########################################################
#one-hot encode entire data frame
##########################################################

dummyset = dummyVars("~ . ", data=traindat)
ohed = na.omit(data.frame(predict(dummyset, newdata=traindat)))
nrow(traindat); nrow(traindat)
ncol(traindat)

xmatall = as.matrix(ohed[, -which(names(ohed) %in% grep("LOSmonths", names(ohed), value=TRUE))])
nrow(xmatall); ncol(xmatall)
head(xmatall[,1:30])
ymatall = as.matrix(ohed[, which(names(ohed) %in% grep("LOSmonths", names(ohed), value=TRUE))])
nrow(ymatall); ncol(ymatall)
head(ymatall)

#free up space on local machine, but keep testdat for later since need to prepare test x and y matrices for prediction
#rm(temp, testdat, traindat, ohed)
rm(temp, traindat, ohed)


##################################################################
#k fold sampler
##################################################################

#save.image("~/MCHS/ELOS/PHIS/keras_ann_full_phis_eplos_2016_2017_cohortV1.RData")


k_fold_sampler = function(FULL_Y_MATRIX, FULL_X_MATRIX, WEIGHT, FOLDS=2){
   if (all.equal(nrow(FULL_X_MATRIX), nrow(FULL_Y_MATRIX))==TRUE) {
     indices_mat = data.frame(index_value = 1:nrow(FULL_X_MATRIX))
      } else {
        stop(paste0("FULL_Y_MATRIX and FULL_X_MATRIX not equal in rows; they must be equal number of rows!\n",
                    "X matrix rows = ", nrow(FULL_X_MATRIX), "\n",
                    "Y matrix rows = ", nrow(FULL_Y_MATRIX), "\n",
                    "Row number differences between X and Y Matrices = ", nrow(FULL_X_MATRIX)-nrow(FULL_Y_MATRIX), " row(s)")
                    )
      }
   indices_mat$fold = sample(1:FOLDS, nrow(indices_mat), replace=TRUE, prob=rep(1/FOLDS, times=FOLDS))
   #ensure that samples in folds are equal to each other (involves some loss of samples, but should do repeated CV folding anyway)
   minimum_fold_nrows = min(data.frame(table(indices_mat$fold))$Freq, na.rm=TRUE)
   #create random fold samples for later reduction to equal folds
   indexes = list()
   xmat_original_fold = list()
   ymat_original_fold = list()
   weight_original_fold = list()
   xmat_min_fold_n = list()
   ymat_min_fold_n = list()
   weight_min_fold_n = list()
   for(i in 1:max(indices_mat$fold)){
     indexes[[i]] = indices_mat$index_value[indices_mat$fold==i]
     xmat_original_fold[[i]] = FULL_X_MATRIX[indexes[[i]],]
     ymat_original_fold[[i]] = FULL_Y_MATRIX[indexes[[i]],]
     weight_original_fold[[i]] = WEIGHT[indexes[[i]],]
     xmat_min_fold_n[[i]] = xmat_original_fold[[i]][1:minimum_fold_nrows,]
     ymat_min_fold_n[[i]] = ymat_original_fold[[i]][1:minimum_fold_nrows,]
     weight_min_fold_n[[i]] = array(as.matrix(weight_original_fold[[i]][1:minimum_fold_nrows]))
   }
 matrices = list(xmat=xmat_min_fold_n, ymat=ymat_min_fold_n, weight=weight_min_fold_n, minimum_samples_fold_n=minimum_fold_nrows)
 return(matrices)  
}

callbacks_list = list(
  callback_early_stopping(monitor = "val_loss", patience=5),
  #callback_model_checkpoint(filepath = "callback_exploration_tryoutV1.h5", monitor = "weighted_acc", save_best_only = TRUE),
  callback_reduce_lr_on_plateau(monitor = "weighted_acc", factor = .1, patience=10))

###################################################
###################################################
###################################################
#3 layer dense network (with FOLDS=2, do not change to workin in keras)
###################################################
###################################################
###################################################

ann_sequential_3_layer = function(ITERATION, FLAG_SET){
  #draw a training a validation fold from data above
  cvfolds = k_fold_sampler(FULL_Y_MATRIX=ymatall, FULL_X_MATRIX=xmatall, WEIGHT=weightests$wt, FOLDS=2)
  #create model
  model = keras_model_sequential() %>%
    layer_dense(units = FLAG_SET$dense_units1, activation = "relu", input_shape = ncol(cvfolds$xmat[[1]])) %>%
    layer_dropout(rate=FLAG_SET$dropout1_proportion) %>%
    layer_activity_regularization(l1=FLAG_SET$L1_penalization1) %>%
    layer_dense(units = FLAG_SET$dense_units2, activation = "relu") %>%
    layer_dropout(rate=FLAG_SET$dropout2_proportion) %>%
    layer_activity_regularization(l1=FLAG_SET$L1_penalization2) %>%
    #layer_dense(units = FLAG_SET$dense_units3, activation = "relu") %>%
    #layer_dropout(rate=FLAG_SET$dropout3_proportion) %>%
    #layer_activity_regularization(l1=FLAG_SET$L1_penalization3) %>%
    layer_dense(units = FLAG_SET$dense_units3, activation="softmax") %>%
    compile(
      optimizer = optimizer_rmsprop(lr=FLAG_SET$learning_rate),
      loss = "categorical_crossentropy",
      metrics = "accuracy",
      weighted_metrics = "accuracy"
      ) 
  summary(model)
  hx = model %>%
    fit(
      cvfolds$xmat[[1]],
      cvfolds$ymat[[1]],
      sample_weight = cvfolds$weight[[1]],
      epochs = FLAG_SET$epochs,
      batch_size = FLAG_SET$batch_size,
      callbacks = callbacks_list,
      validation_data = list(cvfolds$xmat[[2]], cvfolds$ymat[[2]], cvfolds$weight[[2]])
    )
  #save fit history
  hxdat = do.call("cbind.data.frame", hx$metrics)  
  hxdat$repeated_samples = ITERATION
  hxdat$batch_size=hx$params$batch_size
  hxdat$epochs = hx$params$epochs
  hxdat$train_sample_n = hx$params$samples
  hxdat$validation_sample_n = hx$params$validation_samples
return(hxdat)
}

###################################################
###################################################
###################################################
#4 layer dense network (with FOLDS=2, do not change to workin in keras)
###################################################
###################################################
###################################################

ann_sequential_4_layer = function(ITERATION, FLAG_SET){
  #draw a training a validation fold from data above
  cvfolds = k_fold_sampler(FULL_Y_MATRIX=ymatall, FULL_X_MATRIX=xmatall, WEIGHT=weightests$wt, FOLDS=2)
  #create model
  model = keras_model_sequential() %>%
    layer_dense(units = FLAG_SET$dense_units1, activation = "relu", input_shape = ncol(cvfolds$xmat[[1]])) %>%
    layer_dropout(rate=FLAG_SET$dropout1_proportion) %>%
    layer_activity_regularization(l1=FLAG_SET$L1_penalization1) %>%
    layer_dense(units = FLAG_SET$dense_units2, activation = "relu") %>%
    layer_dropout(rate=FLAG_SET$dropout2_proportion) %>%
    layer_activity_regularization(l1=FLAG_SET$L1_penalization2) %>%
    layer_dense(units = FLAG_SET$dense_units3, activation = "relu") %>%
    layer_dropout(rate=FLAG_SET$dropout3_proportion) %>%
    layer_activity_regularization(l1=FLAG_SET$L1_penalization3) %>%
    layer_dense(units = FLAG_SET$dense_units4, activation="softmax") %>%
    compile(
      optimizer = optimizer_rmsprop(lr=FLAG_SET$learning_rate),
      loss = "categorical_crossentropy",
      metrics = "accuracy",
      weighted_metrics = "accuracy"
    ) 
  summary(model)
  hx = model %>%
    fit(
      cvfolds$xmat[[1]],
      cvfolds$ymat[[1]],
      sample_weight = cvfolds$weight[[1]],
      epochs = FLAG_SET$epochs,
      batch_size = FLAG_SET$batch_size,
      callbacks = callbacks_list,
      validation_data = list(cvfolds$xmat[[2]], cvfolds$ymat[[2]], cvfolds$weight[[2]])
    )
  #save fit history
  hxdat = do.call("cbind.data.frame", hx$metrics)  
  hxdat$repeated_samples = ITERATION
  hxdat$batch_size=hx$params$batch_size
  hxdat$epochs = hx$params$epochs
  hxdat$train_sample_n = hx$params$samples
  hxdat$validation_sample_n = hx$params$validation_samples
  return(hxdat)
}

###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
#repeated train; internal-validation splits to make
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################

maximum_times_to_repeat_train_validation = 5

###########################################################
#flag_set1
###########################################################

flag_set1 = flags(
  flag_integer("dense_units1", 90),
  flag_numeric("dropout1_proportion", 0),
  flag_numeric("L1_penalization1", 1),
  flag_integer("dense_units2", 30),
  flag_numeric("dropout2_proportion", 0),
  flag_numeric("L1_penalization2", 1),
  flag_integer("dense_units3", 7),
  flag_integer("epochs", 100),
  flag_integer("batch_size", 200),
  flag_numeric("learning_rate", .001)
)

desc_flag_set1 = data.frame(flag_set1)
desc_flag_set1$set = "set1"

start_time = proc.time()
seqann_set1 = list()
for(i in 1:maximum_times_to_repeat_train_validation){
  seqann_set1[[i]] = ann_sequential_3_layer(ITERATION=i, FLAG_SET=flag_set1)
}
end_time = proc.time()
minutes_run_time = (end_time-start_time)/60
minutes_run_time

set1_hx_summary = do.call("rbind.data.frame", seqann_set1)
set1_hx_summary$set = "set1"

###########################################################
#flag_set2
###########################################################

flag_set2 = flags(
  flag_integer("dense_units1", 90),
  flag_numeric("dropout1_proportion", .5),
  flag_numeric("L1_penalization1", 0),
  flag_integer("dense_units2", 30),
  flag_numeric("dropout2_proportion", .3),
  flag_numeric("L1_penalization2", 0),
  flag_integer("dense_units3", 7),
  flag_integer("epochs", 100),
  flag_integer("batch_size", 200),
  flag_numeric("learning_rate", .001)
)

desc_flag_set2 = data.frame(flag_set2)
desc_flag_set2$set = "set2"

start_time = proc.time()
seqann_set2 = list()
for(i in 1:maximum_times_to_repeat_train_validation){
  seqann_set2[[i]] = ann_sequential_3_layer(ITERATION=i, FLAG_SET=flag_set2)
}
end_time = proc.time()
minutes_run_time = (end_time-start_time)/60
minutes_run_time

set2_hx_summary = do.call("rbind.data.frame", seqann_set2)
set2_hx_summary$set = "set2"


###########################################################
#flag_set3
###########################################################

flag_set3 = flags(
  flag_integer("dense_units1", 90),
  flag_numeric("dropout1_proportion", 0),
  flag_numeric("L1_penalization1", 1),
  flag_integer("dense_units2", 30),
  flag_numeric("dropout2_proportion", 0),
  flag_numeric("L1_penalization2", 1),
  flag_integer("dense_units3", 7),
  flag_integer("epochs", 100),
  flag_integer("batch_size", 200),
  flag_numeric("learning_rate", .001)
)

desc_flag_set3 = data.frame(flag_set3)
desc_flag_set3$set = "set3"

start_time = proc.time()
seqann_set3 = list()
for(i in 1:maximum_times_to_repeat_train_validation){
  seqann_set3[[i]] = ann_sequential_3_layer(ITERATION=i, FLAG_SET=flag_set3)
}
end_time = proc.time()
minutes_run_time = (end_time-start_time)/60
minutes_run_time

set3_hx_summary = do.call("rbind.data.frame", seqann_set3)
set3_hx_summary$set = "set3"

###########################################################
#flag_set4
###########################################################

flag_set4 = flags(
  flag_integer("dense_units1", 90),
  flag_numeric("dropout1_proportion", 0),
  flag_numeric("L1_penalization1", 1),
  flag_integer("dense_units2", 30),
  flag_numeric("dropout2_proportion", 0),
  flag_numeric("L1_penalization2", 0),
  flag_integer("dense_units3", 7),
  flag_integer("epochs", 100),
  flag_integer("batch_size", 200),
  flag_numeric("learning_rate", .001)
)

desc_flag_set4 = data.frame(flag_set4)
desc_flag_set4$set = "set4"

start_time = proc.time()
seqann_set4 = list()
for(i in 1:maximum_times_to_repeat_train_validation){
  seqann_set4[[i]] = ann_sequential_3_layer(ITERATION=i, FLAG_SET=flag_set4)
}
end_time = proc.time()
minutes_run_time = (end_time-start_time)/60
minutes_run_time

set4_hx_summary = do.call("rbind.data.frame", seqann_set4)
set4_hx_summary$set = "set4"

###########################################################
#flag_set5
###########################################################

flag_set5 = flags(
  flag_integer("dense_units1", 100),
  flag_numeric("dropout1_proportion", 0),
  flag_numeric("L1_penalization1", 0),
  flag_integer("dense_units2", 70),
  flag_numeric("dropout2_proportion", 0),
  flag_numeric("L1_penalization2", 0),
  flag_integer("dense_units3", 40),
  flag_numeric("dropout3_proportion", 0),
  flag_numeric("L1_penalization3", 0),
  flag_integer("dense_units4", 7),
  flag_integer("epochs", 100),
  flag_integer("batch_size", 200),
  flag_numeric("learning_rate", .001)
)

desc_flag_set5 = data.frame(flag_set5)
desc_flag_set5$set = "set5"

start_time = proc.time()
seqann_set5 = list()
for(i in 1:maximum_times_to_repeat_train_validation){
  seqann_set5[[i]] = ann_sequential_4_layer(ITERATION=i, FLAG_SET=flag_set5)
}
end_time = proc.time()
minutes_run_time = (end_time-start_time)/60
minutes_run_time

set5_hx_summary = do.call("rbind.data.frame", seqann_set5)
set5_hx_summary$set = "set5"

###########################################################
#flag_set6
###########################################################

flag_set6 = flags(
  flag_integer("dense_units1", 100),
  flag_numeric("dropout1_proportion", .5),
  flag_numeric("L1_penalization1", 0),
  flag_integer("dense_units2", 70),
  flag_numeric("dropout2_proportion", .2),
  flag_numeric("L1_penalization2", 0),
  flag_integer("dense_units3", 40),
  flag_numeric("dropout3_proportion", .1),
  flag_numeric("L1_penalization3", 0),
  flag_integer("dense_units4", 7),
  flag_integer("epochs", 100),
  flag_integer("batch_size", 200),
  flag_numeric("learning_rate", .001)
)

desc_flag_set6 = data.frame(flag_set6)
desc_flag_set6$set = "set6"

start_time = proc.time()
seqann_set6 = list()
for(i in 1:maximum_times_to_repeat_train_validation){
  seqann_set6[[i]] = ann_sequential_4_layer(ITERATION=i, FLAG_SET=flag_set6)
}
end_time = proc.time()
minutes_run_time = (end_time-start_time)/60
minutes_run_time

set6_hx_summary = do.call("rbind.data.frame", seqann_set6)
set6_hx_summary$set = "set6"


###########################################################
#flag_set7
###########################################################

flag_set7 = flags(
  flag_integer("dense_units1", 100),
  flag_numeric("dropout1_proportion", 0),
  flag_numeric("L1_penalization1", 1),
  flag_integer("dense_units2", 70),
  flag_numeric("dropout2_proportion", 0),
  flag_numeric("L1_penalization2", 1),
  flag_integer("dense_units3", 40),
  flag_numeric("dropout3_proportion", 0),
  flag_numeric("L1_penalization3", 1),
  flag_integer("dense_units4", 7),
  flag_integer("epochs", 100),
  flag_integer("batch_size", 200),
  flag_numeric("learning_rate", .001)
)

desc_flag_set7 = data.frame(flag_set7)
desc_flag_set7$set = "set7"

start_time = proc.time()
seqann_set7 = list()
for(i in 1:maximum_times_to_repeat_train_validation){
  seqann_set7[[i]] = ann_sequential_4_layer(ITERATION=i, FLAG_SET=flag_set7)
}
end_time = proc.time()
minutes_run_time = (end_time-start_time)/60
minutes_run_time

set7_hx_summary = do.call("rbind.data.frame", seqann_set7)
set7_hx_summary$set = "set7"



###########################################################
#flag_set8
###########################################################

flag_set8 = flags(
  flag_integer("dense_units1", 100),
  flag_numeric("dropout1_proportion", 0),
  flag_numeric("L1_penalization1", 1),
  flag_integer("dense_units2", 70),
  flag_numeric("dropout2_proportion", 0),
  flag_numeric("L1_penalization2", 0),
  flag_integer("dense_units3", 40),
  flag_numeric("dropout3_proportion", 0),
  flag_numeric("L1_penalization3", 0),
  flag_integer("dense_units4", 7),
  flag_integer("epochs", 100),
  flag_integer("batch_size", 200),
  flag_numeric("learning_rate", .001)
)

desc_flag_set8 = data.frame(flag_set8)
desc_flag_set8$set = "set8"

start_time = proc.time()
seqann_set8 = list()
for(i in 1:maximum_times_to_repeat_train_validation){
  seqann_set8[[i]] = ann_sequential_4_layer(ITERATION=i, FLAG_SET=flag_set8)
}
end_time = proc.time()
minutes_run_time = (end_time-start_time)/60
minutes_run_time

set8_hx_summary = do.call("rbind.data.frame", seqann_set8)
set8_hx_summary$set = "set8"

###########################################################
#flag_set9
###########################################################

flag_set9 = flags(
  flag_integer("dense_units1", 100),
  flag_numeric("dropout1_proportion", .5),
  flag_numeric("L1_penalization1", 0),
  flag_integer("dense_units2", 70),
  flag_numeric("dropout2_proportion", .25),
  flag_numeric("L1_penalization2", 0),
  flag_integer("dense_units3", 40),
  flag_numeric("dropout3_proportion", .25),
  flag_numeric("L1_penalization3", 0),
  flag_integer("dense_units4", 7),
  flag_integer("epochs", 100),
  flag_integer("batch_size", 200),
  flag_numeric("learning_rate", .001)
)

desc_flag_set9 = data.frame(flag_set9)
desc_flag_set9$set = "set9"

start_time = proc.time()
seqann_set9 = list()
for(i in 1:maximum_times_to_repeat_train_validation){
  seqann_set9[[i]] = ann_sequential_4_layer(ITERATION=i, FLAG_SET=flag_set9)
}
end_time = proc.time()
minutes_run_time = (end_time-start_time)/60
minutes_run_time

set9_hx_summary = do.call("rbind.data.frame", seqann_set9)
set9_hx_summary$set = "set9"

###########################################################
#flag_set10
###########################################################

flag_set10 = flags(
  flag_integer("dense_units1", 70),
  flag_numeric("dropout1_proportion", .20),
  flag_numeric("L1_penalization1", 0),
  flag_integer("dense_units2", 50),
  flag_numeric("dropout2_proportion", 0),
  flag_numeric("L1_penalization2", 0),
  flag_integer("dense_units3", 20),
  flag_numeric("dropout3_proportion", 0),
  flag_numeric("L1_penalization3", 0),
  flag_integer("dense_units4", 7),
  flag_integer("epochs", 100),
  flag_integer("batch_size", 200),
  flag_numeric("learning_rate", .001)
)

desc_flag_set10 = data.frame(flag_set10)
desc_flag_set10$set = "set10"

start_time = proc.time()
seqann_set10 = list()
for(i in 1:maximum_times_to_repeat_train_validation){
  seqann_set10[[i]] = ann_sequential_4_layer(ITERATION=i, FLAG_SET=flag_set10)
}
end_time = proc.time()
minutes_run_time = (end_time-start_time)/60
minutes_run_time

set10_hx_summary = do.call("rbind.data.frame", seqann_set10)
set10_hx_summary$set = "set10"


#save.image("~/MCHS/ELOS/PHIS/keras_ann_full_phis_eplos_2016_2017_cohortV1.RData")

###########################################################
###########################################################
###########################################################
#aggregate indicators to assess best accuracy among models
#choose best model there and train on full data from this most accurate model
###########################################################
###########################################################
###########################################################

dput(grep("set[1-9]_hx", ls(), value=TRUE))

hxtunestack = do.call("rbind.data.frame", list(set1_hx_summary, set2_hx_summary, 
                                               set3_hx_summary, set4_hx_summary, 
                                               set5_hx_summary, set6_hx_summary, 
                                               set7_hx_summary, set8_hx_summary,
                                               set9_hx_summary, set10_hx_summary)) %>%
  mutate_if(is.character, as.factor) %>%
  group_by(set, repeated_samples) %>%
  mutate(rownum = row_number()) %>%
  group_by(set, repeated_samples) %>%
  mutate(maxiters = max(rownum)) %>%
  filter(rownum==maxiters) #take the last iteration as the final fit value from which to calculate later loss and accuracy summary statistics

head(hxtunestack)
summary(hxtunestack)
glimpse(hxtunestack)

hxtunestacksum = psych::describeBy(hxtunestack[,-13], hxtunestack$set, mat=TRUE, digits=3) 
hxtunestacksum$metric = gsub('[0-9]+', '', dimnames(hxtunestacksum)[[1]])
hxtunestacksum$set_number = gsub('\\D+','', hxtunestacksum$vars)
hxtunestacksum$groupsetnum = gsub('\\D+','', hxtunestacksum$group1)
hxtunestacksum$set_number_num = as.numeric(as.character(hxtunestacksum$set_number))
hxtunestacksum$Set = paste0("Hyperparameter Tuning Set ", hxtunestacksum$groupsetnum)
hxtunestacksum = hxtunestacksum %>% mutate_if(is.character, as.factor)

table(hxtunestacksum$metric)
head(hxtunestacksum)
names(hxtunestacksum)

hxtunestacksum$Metric = as.character(mapvalues(hxtunestacksum$metric,
                                                   from=c("acc", "batch_size", "epochs", "loss", "lr", "repeated_samples", 
                                                          "train_sample_n", "val_acc", "val_loss", "val_weighted_acc", 
                                                          "validation_sample_n", "weighted_acc"),
                                                   to=c("Accuracy", "Batch Size", "Epochs", "Loss", "Learning Rate", 
                                                        "Repeated Train-Test Validations", 
                                                        "Train N", "Validation Accuracy", "Validation Loss", 
                                                        "Validation Weighted Accuracy", 
                                                        "Validation Sample N", "Weighted Accuracy")
                                                   )
                                            )

subhxtunestacksum = subset(hxtunestacksum, grepl("Accuracy|Weighted Accuracy|Validation Weighted Accuracy|Weighted Accuracy", hxtunestacksum$Metric))


ggplot(subhxtunestacksum) +
    geom_point(aes(x=reorder(Set, set_number_num), y=mean), size=5, colour="dodgerblue3") +
    geom_errorbar(aes(x=reorder(Set, set_number_num), ymin=mean-se, ymax=mean+se), width=0, size=1) +
    #scale_colour_manual(values=c("navy", "firebrick")) +
    facet_wrap(~Metric, nrow=1) +
    coord_flip() + 
    ylab("Accuracy Estimate") +
    xlab("") +
    theme_bw() +
    theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
          strip.text.x = element_text(size=12, face="bold", family="Arial"),
          strip.text.y = element_text(size=12, face="bold", family="Arial"),
          plot.background = element_blank(),
          #axis.text.x=element_blank(),
          axis.text.y=element_text(size=12, family="Arial"),
          panel.grid.major= element_blank(),
          panel.grid.minor = element_blank(),
          legend.position = "bottom")


###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
#build a prediction model from what looks best from internal validation
#predic external validation (test-set)
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################

#currently looks like set 7, so training on full train dataset to create final model for out-of-sample test set predictions
#based on flag_set7, but altering epochs

flag_set10

bestflagset = flag_set10

desc_bestflagset = data.frame(bestflagset)
desc_bestflagset

#create array of weights for full training set
bestmodel_sample_weights = array(as.matrix(weightests$wt))

bestmodelcallbacks_list = list(
  #callback_early_stopping(monitor = "val_loss", patience=5)
  callback_early_stopping(monitor = "loss", patience=10)
  #,
  #callback_model_checkpoint(filepath = "callback_exploration_tryoutV1.h5", monitor = "weighted_acc", save_best_only = TRUE),
  #callback_reduce_lr_on_plateau(monitor = "weighted_acc", factor = .1, patience=5)
  )


start_time = proc.time()
bestmodel = keras_model_sequential() %>%
    layer_dense(units = bestflagset$dense_units1, activation = "relu", input_shape = ncol(xmatall)) %>%
    layer_dropout(rate=bestflagset$dropout1_proportion) %>%
    layer_activity_regularization(l1=bestflagset$L1_penalization1) %>%
    layer_dense(units = bestflagset$dense_units2, activation = "relu") %>%
    layer_dropout(rate=bestflagset$dropout2_proportion) %>%
    layer_activity_regularization(l1=bestflagset$L1_penalization2) %>%
    #layer_dense(units = bestflagset$dense_units3, activation="softmax") %>%
    layer_dense(units = bestflagset$dense_units3, activation = "relu") %>%
    layer_dropout(rate=bestflagset$dropout3_proportion) %>%
    layer_activity_regularization(l1=bestflagset$L1_penalization3) %>%
    layer_dense(units = bestflagset$dense_units4, activation="softmax") %>%
  compile(
      optimizer = optimizer_rmsprop(lr=.0001),
      loss = "categorical_crossentropy",
      metrics = "accuracy",
      weighted_metrics = "accuracy")
  besthx = bestmodel %>%
    fit(
      xmatall,
      ymatall,
      sample_weight = bestmodel_sample_weights,
      epochs = 500,
      batch_size = bestflagset$batch_size,
      callbacks = bestmodelcallbacks_list)
end_time = proc.time()
minutes_run_time = (end_time-start_time)/60
minutes_run_time

besthx
plot(besthx)

# besthx1 = besthx
# besthx2 = besthx
# 
# besthx1
# besthx2

#########################################################################
#prepare test data for input to best prediction model
#########################################################################

#glimpse(testdat)

##########################################################
#SCALE CONTINUOUS VARIABLES BEFORE K-Fold Cross Validation
##########################################################
testdat_age_yrs_descriptives = psych::describe(testdat$age_yrs, na.rm=TRUE)
testdat_age_yrs_descriptives

testdat$age_yrs = min_max_normalization(testdat$age_yrs)
psych::describe(testdat$age_yrs)

##########################################################
#one-hot encode entire data frame
##########################################################

testdummyset = dummyVars("~ . ", data=testdat)
testohed = na.omit(data.frame(predict(testdummyset, newdata=testdat)))
nrow(testohed); nrow(testdat)
ncol(testohed)

test_xmatall = as.matrix(testohed[, -which(names(testohed) %in% grep("LOSmonths", names(testohed), value=TRUE))])
nrow(test_xmatall); ncol(test_xmatall)
head(test_xmatall[,1:30])
test_ymatall = as.matrix(testohed[, which(names(testohed) %in% grep("LOSmonths", names(testohed), value=TRUE))])
nrow(test_ymatall); ncol(test_ymatall)
head(test_ymatall)

#########################################################################
#evaluate model on external validation (test dataset)
#########################################################################

besteval = evaluate(
  bestmodel,
  test_xmatall,
  test_ymatall
)

besteval$loss
besteval$acc
besteval$weighted_acc


#########################################################################
#return predictions on external validation data (test dataset)
#########################################################################

bestpreds_classes = predict_classes(bestmodel, test_xmatall)
bestpreds_probs = predict_proba(bestmodel, test_xmatall)
colnames(bestpreds_probs)=c("a. < 1 Month", "b. 1 Month", "c. 2 Months", "d. 3 Months", "e. 4 Months", "f. 5 Months", "g. 6 Months or more")

table(colnames(bestpreds_probs)[max.col(bestpreds_probs, ties.method="first")])
table(testdat$LOSmonths)

###################################################################
###################################################################
###################################################################
#Discrimination across thresholds of classification for each learner
###################################################################
###################################################################
###################################################################
citation("pROC")

#can't plot, but mean AUC is presented below (Hand and Till, 2001 modeling approach)
#David J. Hand and Robert J. Till (2001). A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems. Machine Learning 45(2), p. 171-186. DOI: 10.1023/A:1010920819831.
multiclass.roc(testdat$LOSmonths, bestpreds_classes)
#likely quite low because some predictions from adjacent categories are close to each other since
#Lenght of Stay was discretized--so, some probabilities for some atients may be
#slightly higher for adjacent groups than the actual group fo the test data. 
#Example, the true group is 2 Month stay (e.g., 62 days) has the predicted probability of .1, while 
#the predicted probability of a 1 months stay while the predicted probabiliy of a 1 month stay (i.e., up to 59 days) is .11, so it 
#is predicted as a 1 month vs. a 2 months stay (true length of stay is 62 days)

#perfect description of interpretation of discrimination underfitting/overfitting and under-estimation vs over-estimation on pg 30
#ftp://ftp.esat.kuleuven.be/pub/SISTA/ida/reports/14-224.pdf 

discrimination_auroc_estimation = function(BINARY_TARGET, MODEL_PREDICTION, MODEL_DESCRIPTION_TEXT=NA){
  roco = plot.roc(BINARY_TARGET, MODEL_PREDICTION, percent=TRUE, ci=TRUE, col="navy", lwd=4, print.auc=TRUE, print.auc.y=50)
  #Note that [-1] removes -Inf, 100, and 0 from thresholds, sensitivities, and specificities, respectively--allows for equal rows with response and predictor vectors
  #rocdat = cbind.data.frame(response=roco$response, prediction=roco$predictor, sensitivities=roco$sensitivities[-1], specificities=roco$specificities[-1], thresholds=roco$thresholds[-1], Description=rep(MODEL_DESCRIPTION_TEXT, times=length(roco$sensitivities[-1])))
  rocdat = cbind.data.frame(sensitivities=roco$sensitivities[-1], specificities=roco$specificities[-1], thresholds=roco$thresholds[-1], Description=rep(MODEL_DESCRIPTION_TEXT, times=length(roco$sensitivities[-1])))
  rocdat$AUC_formatted = roco$auc
  rocdat$AUC_CI_formatted = paste0(round(roco$auc, digits=2), "% CI: ", round(as.numeric(roco$ci)[1], digits=2), "% to ", round(as.numeric(roco$ci)[3], digits=2),"%") 
  rocdat$auc = as.numeric(roco$ci)[2]
  rocdat$aucll = as.numeric(roco$ci)[1]
  rocdat$aucul = as.numeric(roco$ci)[3]
  rocdat$Full_Description = paste0(rocdat$Description, " AUC: ", rocdat$AUC_CI_formatted)
  rocretlist = list(rocdat=rocdat, roc_object=roco)
  return(rocretlist)
}

rocker0 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,1], MODEL_PREDICTION=bestpreds_probs[,1], MODEL_DESCRIPTION_TEXT="a. <1 Month")
rocker1 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,2], MODEL_PREDICTION=bestpreds_probs[,2], MODEL_DESCRIPTION_TEXT="b. 1 to 2 Months")
rocker2 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,3], MODEL_PREDICTION=bestpreds_probs[,3], MODEL_DESCRIPTION_TEXT="c. 2 to 3 Months")
rocker3 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,4], MODEL_PREDICTION=bestpreds_probs[,4], MODEL_DESCRIPTION_TEXT="d. 3 to 4 Months")
rocker4 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,5], MODEL_PREDICTION=bestpreds_probs[,5], MODEL_DESCRIPTION_TEXT="e. 4 to 5 Months")
rocker5 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,6], MODEL_PREDICTION=bestpreds_probs[,6], MODEL_DESCRIPTION_TEXT="f. 5 to 6 Months")
rocker6 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,7], MODEL_PREDICTION=bestpreds_probs[,7], MODEL_DESCRIPTION_TEXT="g. 6+ Months")

rocdats = rbind.data.frame(rocker0$rocdat, rocker1$rocdat, 
                           rocker2$rocdat, rocker3$rocdat, 
                           rocker4$rocdat, rocker5$rocdat, 
                           rocker6$rocdat)

head(rocdats)
summary(rocdats)

rocker0$roc_object

discrimsplot = ggplot(rocdats) +
  geom_abline(intercept=0, slope=1, size=1) +
  geom_line(data=rocdats, aes(x=100-specificities, y=sensitivities, group=Full_Description, colour=Full_Description, linetype=Full_Description), size=1.25) +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  scale_linetype_discrete(name="Length of Stay") +
  scale_colour_manual(values=c("gold2", "orange", "orangered", "red", "firebrick1", "firebrick", "firebrick4"), name="Length of Stay") +
  #facet_wrap(~Full_Description, nrow=1) +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,1)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,1)) +
  scale_x_continuous(limits=c(0, 100)) +
  scale_y_continuous(limits=c(0, 100)) +
  ggtitle("Full ANN Model Prediction") +
  ylab("Sensitivity (%)") +
  xlab("1-Specificity (%)") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.minor = element_blank(),
        legend.position = "right")
discrimsplot
ggsave("complete_feature_space_EPLOS_discrimination.pdf", discrimsplot, width=8, height=6, scale=1.2)


#Note that sample sizes result in statistically significant differences even with very small effect size differences between classifiers
#This is the Delong Test to compare AUROCs
#test differences in AUCs between different learners
roc.test(rocker0$roc_object, rocker1$roc_object)
roc.test(rocker0$roc_object, rocker2$roc_object)
roc.test(rocker0$roc_object, rocker3$roc_object)
roc.test(rocker0$roc_object, rocker4$roc_object)
roc.test(rocker0$roc_object, rocker5$roc_object)
roc.test(rocker0$roc_object, rocker6$roc_object)
roc.test(rocker1$roc_object, rocker2$roc_object)
roc.test(rocker1$roc_object, rocker3$roc_object)
roc.test(rocker1$roc_object, rocker4$roc_object)
roc.test(rocker1$roc_object, rocker5$roc_object)
roc.test(rocker1$roc_object, rocker6$roc_object)
roc.test(rocker2$roc_object, rocker3$roc_object)
roc.test(rocker2$roc_object, rocker4$roc_object)
roc.test(rocker2$roc_object, rocker5$roc_object)
roc.test(rocker2$roc_object, rocker6$roc_object)
roc.test(rocker3$roc_object, rocker4$roc_object)
roc.test(rocker3$roc_object, rocker5$roc_object)
roc.test(rocker3$roc_object, rocker6$roc_object)
roc.test(rocker4$roc_object, rocker5$roc_object)
roc.test(rocker4$roc_object, rocker6$roc_object)
roc.test(rocker5$roc_object, rocker6$roc_object)

# #compare = roc.test(rocplot1, rocplot2)
# #compp = ifelse(round(as.numeric(compare$p.value), digits=3)<.001, .001, round(as.numeric(compare$p.value), digits=3))
# #text(30, 30, labels=paste0("Delong Test = ", round(compare$statistic, digits=2), ", p<", compp), cex=1.2, col="black")


###################################################################
###################################################################
###################################################################
#Calibrations of classification for each category of the learner
###################################################################
###################################################################
###################################################################

hosmer_lemeshow_glm_function = function(TARGET, PREDICTION, GROUPS_TO_CUT_BY, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05, PREDICTION_MODEL_DESCRIPTION_TEXT=""){
  require(ResourceSelection)
  hltest = ResourceSelection::hoslem.test(TARGET, PREDICTION, g=GROUPS_TO_CUT_BY)
  #cut predictions and run through logistic regression
  cut_pred = cut(PREDICTION, breaks=unique(quantile(PREDICTION, probs=c(0, (1:GROUPS_TO_CUT_BY)/GROUPS_TO_CUT_BY))), include.lowest=TRUE, labels=FALSE)
  HLpredfulldat = na.omit(cbind.data.frame(target = TARGET, original_pred = PREDICTION, HLgroups = factor(cut_pred)))
  #predict mean response per group with standard error
  HLlogistic = glm(target ~ HLgroups, data=HLpredfulldat, family="binomial")
  HLpreddat = data.frame(HLpredfulldat %>% group_by(HLgroups) %>% summarize(original_pred_group_mean = mean(original_pred, na.rm=TRUE)))
  temp_preds = predict(HLlogistic, newdata=HLpreddat, se.fit = TRUE)
  HLpreddat$group_pred = as.vector(temp_preds$fit)
  HLpreddat$group_se = as.vector(temp_preds$se.fit)
  HLpreddat$group_p = boot::inv.logit(HLpreddat$group_pred)
  HLpreddat$group_ll = boot::inv.logit(HLpreddat$group_pred + HLpreddat$group_se*qnorm(P_VALUE_FOR_CONFIDENCE_INTERVAL/2))
  HLpreddat$group_ul = boot::inv.logit(HLpreddat$group_pred + HLpreddat$group_se*qnorm(1-(P_VALUE_FOR_CONFIDENCE_INTERVAL/2)))
  HLpreddat$Model = rep(PREDICTION_MODEL_DESCRIPTION_TEXT, times=nrow(HLpreddat))
  HLpredintegrate = data.frame(HLpreddat
                               #, 
                               #quantile_range = dimnames(as.data.frame.matrix(hltest$observed))[[1]], 
                               #quantile_range = levels(cut(PREDICTION, breaks=unique(quantile(PREDICTION, probs=c(0, (1:GROUPS_TO_CUT_BY)/GROUPS_TO_CUT_BY))), include.lowest=TRUE))
  )
  HL_counts = data.frame(as.data.frame.matrix(hltest$observed), as.data.frame.matrix(hltest$expected), n = rowSums(as.data.frame.matrix(hltest$observed)))
  rownames(HLpredintegrate) = c() #remove rownames
  HLretlist = list(predictions = HLpredintegrate, HL_test = hltest, HL_counts = HL_counts) 
  return(HLretlist)
}


#choose 20 because that gives at least a marginal average of 5 patients per the 5 to 6 month group, smallest
#observed probability distribution
table(testdat$LOSmonths)

loscat_hosmers = list()
loscat_hosmer_predictions = list()
for(i in 1:ncol(test_ymatall)){
  #run hosmer and lemeshow groupings
  loscat_hosmers[[i]] = hosmer_lemeshow_glm_function(TARGET=test_ymatall[,i], PREDICTION=bestpreds_probs[,i], 
                                                     GROUPS_TO_CUT_BY=20, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                                     PREDICTION_MODEL_DESCRIPTION_TEXT = "Full Feature Space")
  loscat_hosmers[[i]]$predictions$sample = i
  #create datasets of predictions
  loscat_hosmer_predictions[[i]] = loscat_hosmers[[i]]$predictions
}
loscat_hosmer_preds_dat = do.call("rbind.data.frame", loscat_hosmer_predictions)
loscat_hosmer_preds_dat$sample = factor(loscat_hosmer_preds_dat$sample)
head(loscat_hosmer_preds_dat)

dput(levels(loscat_hosmer_preds_dat$sample))
dput(levels(rocdats$Description))
loscat_hosmer_preds_dat$LOS = factor(as.character(mapvalues(loscat_hosmer_preds_dat$sample,
                                                            from=c("1", "2", "3", "4", "5", "6", "7"),
                                                            to=c("a. <1 Month", "b. 1 to 2 Months", "c. 2 to 3 Months", "d. 3 to 4 Months", 
                                                                 "e. 4 to 5 Months", "f. 5 to 6 Months", "g. 6+ Months"))))

#calibrations plot 
calibsplot = ggplot(loscat_hosmer_preds_dat) +
  geom_abline(intercept=0, slope=1, size=.5) +
  #geom_hline(yintercept=0, colour="black", size=.3) +
  geom_line(aes(x=original_pred_group_mean, y=group_p, group=LOS, colour=LOS), size=3) +
  #geom_point(aes(x=original_pred_group_mean, y=group_p, group=sample, colour=sample), size=2) +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  scale_colour_manual(values=c("gold2", "orange", "orangered", "red", "firebrick1", "firebrick", "firebrick4"), name="Model") +
  facet_wrap(~LOS, nrow=1) +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,0)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,0)) +
  scale_x_continuous(limits=c(0, 1), breaks = seq(0, 1, by=.2)) +
  scale_y_continuous(limits=c(0, 1), breaks = seq(0, 1, by=.1)) +
  ggtitle("Full ANN Model Prediction") +
  ylab("Observed Probability") +
  xlab("Prediction") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=8),
        axis.text.y=element_text(size=10),
        #panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "none")
calibsplot
ggsave("complete_feature_space_EPLOS_calibration.pdf", calibsplot, width=11, height=6, scale=1.2)

##############################################################
#SAVE IMAGE
##############################################################

#save.image("~/MCHS/ELOS/PHIS/keras_ann_full_phis_eplos_2016_2017_cohortV1.RData")


##############################################################
##############################################################
##############################################################
#assess variable association with optimal classifiers
##############################################################
##############################################################
##############################################################

glimpse(test_xmatall)
dimnames(test_xmatall)[[2]]

table(bestpreds_classes)
psych::describe(bestpreds_probs)

table(colnames(bestpreds_probs)[max.col(bestpreds_probs, ties.method="first")])
table(testdat$LOSmonths)

ncol(bestpreds_probs)
ncol(test_xmatall)

exclusions_comparisons = list()
all_exclusion_comparisons = list()
for (j in 1:ncol(bestpreds_probs)){
  for(i in c(60,65,102,106,123,129,144,154)){
    exclusions_comparisons[[i]] = psych::describeBy(bestpreds_probs[,j], test_xmatall[,i], mat = TRUE, digits=3)
    exclusions_comparisons[[i]]$varlist_num = i
    exclusions_comparisons[[i]]$var_description = dimnames(test_xmatall)[[2]][i]
  }
 all_exclusion_comparisons[[j]] = do.call("rbind.data.frame", exclusions_comparisons)
 all_exclusion_comparisons[[j]]$outcome_num = j
 all_exclusion_comparisons[[j]]$outcome_name = dimnames(bestpreds_probs)[[2]][j]
}

allexcldat = do.call("rbind.data.frame", all_exclusion_comparisons)
names(allexcldat)
#View(subset(allexcldat, select=c(outcome_name, var_description, group1, n, mean, sd, median)))


#check association between each feature and optimal prediction
#excluding a few items from above that did not converge under betareg assumptions and numerical optimization routines available
#zero-inflated and bound beta distribution through JAGS could take days to estimate for some of these covariates based on documented performance times
starttime = proc.time()
betareg_mod = list()
ascdat = list()
ascdatall = list()
for (j in 1:ncol(bestpreds_probs)){
  for (i in c(1:59,61:64,66:101,103:105,107:122,124:128,130:143,145:153,155:178)){
    print(paste0("################## currently running outcome number ", j, " WITH predictor number ", i, " ##################"))
    print(paste0("################## currently running outcome =  ", dimnames(bestpreds_probs)[[2]][j], " WITH predictor = ", dimnames(test_xmatall)[[2]][i], " ##################"))
    tryCatch({
    #betareg::betareg.control(fstol = 1e-4)
    betareg_mod = suppressWarnings(betareg::betareg(bestpreds_probs[,j]~test_xmatall[,i], link="logit")) 
    ascdat[[i]] = data.frame(feature = dimnames(test_xmatall)[[2]][i],
                    rho = cor(bestpreds_probs[,j], test_xmatall[,i], method="spearman"),
                    #beta = summary(glm(boot::logit(bestpreds_probs[,j])~test_xmatall[,i], family=gaussian))$coefficients[2,1])
                    beta = summary(glm(VGAM::cloglog(bestpreds_probs[,j], bvalue=1e-45)~test_xmatall[,i], family=gaussian))$coefficients[2,1],
                    betaregOR = exp(betareg_mod$coefficients$mean[2])
    )
    }
    )
    ascdat[[i]]$rho_sign = sign(ascdat[[i]]$rho)
    ascdat[[i]]$beta_sign = sign(ascdat[[i]]$beta)
  }
  ascdatall[[j]] = do.call("rbind.data.frame", ascdat)
  ascdatall[[j]]$Outcome = dimnames(bestpreds_probs)[[2]][j]
}
endtime = proc.time()
minutes_run_time = (endtime-starttime)/60
minutes_run_time

corasc = do.call("rbind.data.frame", ascdatall) 

corasc$neg_rho = ifelse(corasc$rho_sign==-1, corasc$rho, NA)
corasc$pos_rho = ifelse(corasc$rho_sign==1, corasc$rho, NA)
corasc$neg_beta = ifelse(corasc$beta_sign==-1, corasc$beta, NA)
corasc$pos_beta = ifelse(corasc$beta_sign==1, corasc$beta, NA)

#log ORs for effect size beta sullivan scoring
corasc$logbetareg = log(corasc$betaregOR)
corasc$neg_logbetareg = ifelse(corasc$logbetareg<0, corasc$logbetareg, NA)
corasc$pos_logbetareg = ifelse(corasc$logbetareg>0, corasc$logbetareg, NA)

corasc = corasc %>%
  #arrange(Outcome, -beta) %>%
  arrange(Outcome, -logbetareg) %>%
  group_by(Outcome) %>%
  mutate(order=rank(-logbetareg, ties="first")) %>%
  #mutate(order = rank(-beta, ties="first")) %>%
  arrange(Outcome, order)

corasc[corasc$feature %in% grep("P229", levels(corasc$feature), value=TRUE),]

summary(corasc)

magplot = ggplot(data=corasc, group=1) +
  geom_point(aes(x=reorder(feature, logbetareg), y=logbetareg, colour=logbetareg), size=1) +
  geom_errorbar(aes(x=reorder(feature, logbetareg), ymin=0, ymax=pos_logbetareg, colour=logbetareg), size=.5) +
  geom_errorbar(aes(x=reorder(feature, logbetareg), ymin=neg_logbetareg, ymax=0, colour=logbetareg), size=.5) +
  scale_colour_gradient2(low="navy", mid="grey90", high="firebrick", name="Beta Regression Coefficient") +
  #scale_linetype_discrete(name="Length of Stay") +
  #scale_colour_manual(values=c("gold2", "orange", "orangered", "red", "firebrick1", "firebrick", "firebrick4"), name="Length of Stay") +
  facet_wrap(~Outcome, nrow=1) +
  coord_flip() +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,1)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,1)) +
#  scale_x_continuous(limits=c(0, 100)) +
#  scale_y_continuous(limits=c(0, 100)) +
  ggtitle("") +
  ylab("Magnitude of Association between Feature and Outcome") +
  xlab("Feature") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=3),
        #panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "bottom")
magplot
ggsave("EPLOS_magnitude_of_effects_beta_ANN_classifier.pdf", magplot, width=11, height=8, scale=1.2)

##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
#Sullivan point scoring adaptation, but not using Betas, using Rhos--seem to perform better in terms of classifier prediction
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################

#get names of DXs and PXs ####################################
dxnames = read.csv("DX_features_with_at_least_20_neonates_with_EPLOS.csv", header=TRUE)[,-1]
pxnames = read.csv("PX_features_with_at_least_20_neonates_with_EPLOS.csv", header=TRUE)[,-1]
head(dxnames)
head(pxnames)

head(corasc)

namedfeats = merge(
  merge(corasc, subset(dxnames, select=c(icdcode, icd)), by.x=c("feature"), by.y=c("icdcode"), all.x=TRUE), 
  subset(pxnames, select=c(pxicdcode, pxicd)), by.x=c("feature"), by.y=c("pxicdcode"), all.x=TRUE)
dput(names(namedfeats))
featdat = subset(namedfeats, select=c("Outcome", "feature", 
                                      "betaregOR", "logbetareg", "neg_logbetareg", "pos_logbetareg", 
                                      "rho", "rho_sign", "neg_rho", "pos_rho",
                                      "beta", "beta_sign", 
                                      "order", "icd", "pxicd"))
featdat$label = coalesce(as.character(featdat$icd), as.character(featdat$pxicd))

featdat$code_type = gsub( "_.*$", "", featdat$feature)


featdat1 = featdat %>%
  group_by(Outcome, code_type) %>%  
  #mutate(Strength_Ordering = rank(-abs(beta), ties="first")) %>%
  mutate(Strength_Ordering = rank(-logbetareg, ties="first")) %>% #takes the highest risk for extended outcomes
  arrange(Outcome, code_type, Strength_Ordering)

#View(subset(featdat1, select=c(feature, rho, Outcome, code_type, Strength_Ordering, label)))

#######################################################################
#selection of top DXs and PXs per multinomia outcome category
#######################################################################

dput(levels(factor(corasc$Outcome)))

INCLUSION_THRESHOLD = 10

top0 = subset(featdat1, Outcome=="a. < 1 Month" & Strength_Ordering<=INCLUSION_THRESHOLD)
top1 = subset(featdat1, Outcome=="b. 1 Month" & Strength_Ordering<=INCLUSION_THRESHOLD)
top2 = subset(featdat1, Outcome=="c. 2 Months" & Strength_Ordering<=INCLUSION_THRESHOLD)
top3 = subset(featdat1, Outcome=="d. 3 Months" & Strength_Ordering<=INCLUSION_THRESHOLD)
top4 = subset(featdat1, Outcome=="e. 4 Months" & Strength_Ordering<=INCLUSION_THRESHOLD)
top5 = subset(featdat1, Outcome=="f. 5 Months" & Strength_Ordering<=INCLUSION_THRESHOLD)
top6 = subset(featdat1, Outcome=="g. 6 Months or more" & Strength_Ordering<=INCLUSION_THRESHOLD)


##############################################################
#Sullivan scoring format
##############################################################

sullivan_scoring_formatter = function(DATA, DIVISOR=1){
  #DATA$minbeta_gt0 = min(ifelse(DATA$logbetareg>0, DATA$logbetareg, NA), na.rm=TRUE)
  #DATA$maxbeta_lt0 = max(ifelse(DATA$logbetareg<0, DATA$logbetareg, NA), na.rm=TRUE)
  #DATA$sullivan_ref = ifelse(DATA$logbetareg>0, DATA$minbeta_gt0, DATA$maxbeta_lt0)
  DATA$absbeta = abs(DATA$logbetareg)
  DATA$sullivan_ref = min(DATA$absbeta) #sullivan reference is the minimum absolute value of covariates observed in selection
  DATA$raw_sullivan = (DATA$beta/DATA$sullivan_ref)*DATA$beta_sign
  #DATA$raw_sullivan = DATA$logbetareg/DATA$sullivan_ref
  DATA$sullivan_points = round(DATA$raw_sullivan/DIVISOR, digits=0)
  dat = DATA
  return(dat)
}

ss0 = sullivan_scoring_formatter(top0, DIVISOR = 1)
ss1 = sullivan_scoring_formatter(top1, DIVISOR = 1)
ss2 = sullivan_scoring_formatter(top2, DIVISOR = 1)
ss3 = sullivan_scoring_formatter(top3, DIVISOR = 1)
ss4 = sullivan_scoring_formatter(top4, DIVISOR = 1)
ss5 = sullivan_scoring_formatter(top5, DIVISOR = 1)
ss6 = sullivan_scoring_formatter(top6, DIVISOR = 1)

subset(ss0, select=c("Outcome", "code_type", "label", "Strength_Ordering",  "logbetareg", "sullivan_ref", "raw_sullivan", "sullivan_points"))
subset(ss1, select=c("Outcome", "code_type", "label", "Strength_Ordering",  "logbetareg", "sullivan_ref", "raw_sullivan", "sullivan_points"))
subset(ss2, select=c("Outcome", "code_type", "label", "Strength_Ordering",  "logbetareg", "sullivan_ref", "raw_sullivan", "sullivan_points"))
subset(ss3, select=c("Outcome", "code_type", "label", "Strength_Ordering",  "logbetareg", "sullivan_ref", "raw_sullivan", "sullivan_points"))
subset(ss4, select=c("Outcome", "code_type", "label", "Strength_Ordering",  "logbetareg", "sullivan_ref", "raw_sullivan", "sullivan_points"))
subset(ss5, select=c("Outcome", "code_type", "label", "Strength_Ordering",  "logbetareg", "sullivan_ref", "raw_sullivan", "sullivan_points"))
subset(ss6, select=c("Outcome", "code_type", "label", "Strength_Ordering",  "logbetareg", "sullivan_ref", "raw_sullivan", "sullivan_points"))


#############################################################
#return scoring algorithm points
#############################################################

sullscoredat = subset(rbind.data.frame(ss0, ss1, ss2, ss3, ss4, ss5, ss6), 
                      select=c(Outcome, feature, label, code_type, logbetareg, Strength_Ordering, sullivan_points)) %>%
  arrange(desc(Outcome), code_type, desc(logbetareg))

write.csv(sullscoredat, "sullivan_scoring_by_outcome_and_predictive_feature.csv")


#############################################################
#prepare scoring on all 2016-2017 CHA PHIS cohort data for estimates of percentile rank of resulting scores
#############################################################

featvars = data.frame(feature = dimnames(test_xmatall)[[2]])
featvars1 = merge(featvars, subset(dxnames, select=c(icdcode, icd)), by.x=c("feature"), by.y=c("icdcode"), all.x=TRUE)
featvars2 = merge(featvars1, subset(pxnames, select=c(pxicdcode, pxicd)), by.x=c("feature"), by.y=c("pxicdcode"), all.x=TRUE)
featvars2$label = coalesce(as.character(featvars2$pxicd), as.character(featvars2$icd))
fv3 = merge(subset(featvars2, select=c(label, feature)), subset(ss0, select=c(feature, sullivan_points)), by.x=c("feature"), by.y=c("feature"), all.x=TRUE)
colnames(fv3) = c("feature", "label", "lt1")
fv4 = merge(fv3, subset(ss1, select=c(feature, sullivan_points)), by.x=c("feature"), by.y=c("feature"), all.x=TRUE)
colnames(fv4) = c("feature", "label", "lt1", "lt2")
fv5 = merge(fv4, subset(ss2, select=c(feature, sullivan_points)), by.x=c("feature"), by.y=c("feature"), all.x=TRUE)
colnames(fv5) = c("feature", "label", "lt1", "lt2", "lt3")
fv6 = merge(fv5, subset(ss3, select=c(feature, sullivan_points)), by.x=c("feature"), by.y=c("feature"), all.x=TRUE)
colnames(fv6) = c("feature", "label", "lt1", "lt2", "lt3", "lt4")
fv7 = merge(fv6, subset(ss4, select=c(feature, sullivan_points)), by.x=c("feature"), by.y=c("feature"), all.x=TRUE)
colnames(fv7) = c("feature", "label", "lt1", "lt2", "lt3", "lt4", "lt5")
fv8 = merge(fv7, subset(ss5, select=c(feature, sullivan_points)), by.x=c("feature"), by.y=c("feature"), all.x=TRUE)
colnames(fv8) = c("feature", "label", "lt1", "lt2", "lt3", "lt4", "lt5", "lt6")
fv9 = merge(fv8, subset(ss6, select=c(feature, sullivan_points)), by.x=c("feature"), by.y=c("feature"), all.x=TRUE)
colnames(fv9) = c("feature", "label", "lt1", "lt2", "lt3", "lt4", "lt5", "lt6", "gt6")

fv9[is.na(fv9)] = 0

#check whether conformable
ncol(test_xmatall)
nrow(fv9)

lt1score = test_xmatall %*% as.matrix(subset(fv9, select=c(lt1)))
lt2score = test_xmatall %*% as.matrix(subset(fv9, select=c(lt2)))
lt3score = test_xmatall %*% as.matrix(subset(fv9, select=c(lt3)))
lt4score = test_xmatall %*% as.matrix(subset(fv9, select=c(lt4)))
lt5score = test_xmatall %*% as.matrix(subset(fv9, select=c(lt5)))
lt6score = test_xmatall %*% as.matrix(subset(fv9, select=c(lt6)))
gt6score = test_xmatall %*% as.matrix(subset(fv9, select=c(gt6)))

sullscoremat = as.matrix(cbind.data.frame(lt1score, lt2score, lt3score, lt4score, lt5score, lt6score, gt6score))
summary(sullscoremat)
psych::describe(sullscoremat)

###########################################################################
###########################################################################
###########################################################################
#Sullivan adapted scoring for Rho discriminations
###########################################################################
###########################################################################
###########################################################################

dev.off()
sullroc0 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,1], MODEL_PREDICTION=sullscoremat[,1], MODEL_DESCRIPTION_TEXT="a. <1 Month")
sullroc1 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,2], MODEL_PREDICTION=sullscoremat[,2], MODEL_DESCRIPTION_TEXT="b. 1 to 2 Months")
sullroc2 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,3], MODEL_PREDICTION=sullscoremat[,3], MODEL_DESCRIPTION_TEXT="c. 2 to 3 Months")
sullroc3 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,4], MODEL_PREDICTION=sullscoremat[,4], MODEL_DESCRIPTION_TEXT="d. 3 to 4 Months")
sullroc4 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,5], MODEL_PREDICTION=sullscoremat[,5], MODEL_DESCRIPTION_TEXT="e. 4 to 5 Months")
sullroc5 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,6], MODEL_PREDICTION=sullscoremat[,6], MODEL_DESCRIPTION_TEXT="f. 5 to 6 Months")
sullroc6 = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,7], MODEL_PREDICTION=sullscoremat[,7], MODEL_DESCRIPTION_TEXT="g. 6+ Months")


sullrocdats = rbind.data.frame(sullroc0$rocdat, sullroc1$rocdat, 
                           sullroc2$rocdat, sullroc3$rocdat, 
                           sullroc4$rocdat, sullroc5$rocdat, 
                           sullroc6$rocdat)

head(sullrocdats)
summary(sullrocdats)

rocdats$Score = "Full ANN Model Prediction"
sullrocdats$Score = "Simplified Sullivan Point Scoring"

head(sullrocdats)
#manually add in asymptote in visual
aggsullrocs = sullrocdats %>%
  group_by_("Description", "AUC_formatted", "AUC_CI_formatted", "auc", "aucll", "aucul", 
            "Full_Description", "Score") %>%
  summarize(sensitivities = 99,
            specificities = 1,
            thresholds = 0)

head(aggsullrocs)
reorderaggsullrocs = subset(aggsullrocs, select=dput(names(sullrocdats)))

sullrocsrev = rbind.data.frame(sullrocdats, reorderaggsullrocs)

sulldiscrimsplot = ggplot(sullrocsrev) +
  geom_abline(intercept=0, slope=1, size=1) +
  geom_line(aes(x=100-specificities, y=sensitivities, group=Full_Description, colour=Full_Description, linetype=Full_Description), size=1.25) +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  scale_linetype_discrete(name="Length of Stay") +
  scale_colour_manual(values=c("lightblue", "dodgerblue1", "dodgerblue2", "dodgerblue", "dodgerblue3", "dodgerblue4", "navy"), name="Length of Stay") +
  #facet_wrap(~Full_Description, nrow=1) +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,1)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,1)) +
  scale_x_continuous(limits=c(0, 100)) +
  scale_y_continuous(limits=c(0, 100)) +
  ggtitle("Simplified Sullivan Point Scoring") +
  ylab("Sensitivity (%)") +
  xlab("1-Specificity (%)") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.minor = element_blank(),
        legend.position = "right")
sulldiscrimsplot
ggsave("simplified_sullivan_feature_space_EPLOS_discrimination.pdf", sulldiscrimsplot, width=8, height=6, scale=1.2)

ardiscrimplots = gridExtra::grid.arrange(discrimsplot, sulldiscrimsplot, ncol=1)
ardiscrimplots

ggsave("disciminations_EPLOS_combined.pdf", ardiscrimplots, width=8, height=8, scale=1.2)

###########################################################################
###########################################################################
###########################################################################
#Sullivan adapted scoring for rhos calibrations
###########################################################################
###########################################################################
###########################################################################

sull_score_logit_models = list()
sull_score_cloglog_models = list()
sull_score_logit_predictions = list()
sull_score_cloglog_predictions = list()
for(i in 1:ncol(sullscoremat)){
  sull_score_logit_models[[i]] = glm(test_ymatall[,i]~sullscoremat[,i], family=binomial(link="logit"))
  sull_score_cloglog_models[[i]] = glm(test_ymatall[,i]~sullscoremat[,i], family=binomial(link="cloglog"))
  sull_score_logit_predictions[[i]] = predict(sull_score_logit_models[[i]], type="response")
  sull_score_cloglog_predictions[[i]] = predict(sull_score_cloglog_models[[i]], type="response")
}

#run summary statistics on preictions with test data
for(i in 1:ncol(sullscoremat)){
  #print(summary(sull_score_logit_models[[i]]))
  #print(summary(sull_score_cloglog_models[[i]]))
  print(cbind.data.frame(AIC(sull_score_logit_models[[i]]), AIC(sull_score_cloglog_models[[i]])))
}

sull_score_logit_preds_dat = do.call("cbind.data.frame", sull_score_logit_predictions)
colnames(sull_score_logit_preds_dat) = c("1", "2", "3", "4", "5", "6", "7")

sullcat_hosmers = list()
sullcat_hosmer_predictions = list()
for(i in 1:ncol(test_ymatall)){
  #run hosmer and lemeshow groupings
  sullcat_hosmers[[i]] = hosmer_lemeshow_glm_function(TARGET=test_ymatall[,i], PREDICTION=sull_score_logit_predictions[[i]], 
                                                      GROUPS_TO_CUT_BY=20, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                                      PREDICTION_MODEL_DESCRIPTION_TEXT = "Simplified Score")
  sullcat_hosmers[[i]]$predictions$sample = i
  #create datasets of predictions
  sullcat_hosmer_predictions[[i]] = sullcat_hosmers[[i]]$predictions
}
sullcat_hosmer_preds_dat = do.call("rbind.data.frame", sullcat_hosmer_predictions)
sullcat_hosmer_preds_dat$sample = factor(sullcat_hosmer_preds_dat$sample)
head(sullcat_hosmer_preds_dat)

dput(levels(sullcat_hosmer_preds_dat$sample))
dput(levels(rocdats$Description))
sullcat_hosmer_preds_dat$LOS = factor(as.character(mapvalues(sullcat_hosmer_preds_dat$sample,
                                                             from=c("1", "2", "3", "4", "5", "6", "7"),
                                                             to=c("a. <1 Month", "b. 1 to 2 Months", "c. 2 to 3 Months", "d. 3 to 4 Months", 
                                                                  "e. 4 to 5 Months", "f. 5 to 6 Months", "g. 6+ Months"))))

#calibrations plot 
sullcalibsplot = ggplot(sullcat_hosmer_preds_dat) +
  geom_abline(intercept=0, slope=1, size=.5) +
  #geom_hline(yintercept=0, colour="black", size=.3) +
  geom_line(aes(x=original_pred_group_mean, y=group_p, group=LOS, colour=LOS), size=3) +
  #geom_point(aes(x=original_pred_group_mean, y=group_p, group=sample, colour=sample), size=2) +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  scale_colour_manual(values=c("lightblue", "dodgerblue1", "dodgerblue2", "dodgerblue", "dodgerblue3", "dodgerblue4", "navy"), name="Length of Stay") +
  facet_wrap(~LOS, nrow=1) +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,0)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,0)) +
  scale_x_continuous(limits=c(0, 1), breaks = seq(0, 1, by=.2)) +
  scale_y_continuous(limits=c(0, 1), breaks = seq(0, 1, by=.1)) +
  ggtitle("Simplified Sullivan Point Scoring") +
  ylab("Observed Probability") +
  xlab("Prediction") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=8),
        axis.text.y=element_text(size=10),
        #panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "none")
sullcalibsplot
ggsave("sullivan_EPLOS_calibration.pdf", plot=sullcalibsplot, width=11, height=6, scale=1.2)

arcalibsplots = gridExtra::grid.arrange(calibsplot, sullcalibsplot, ncol=1)
arcalibsplots
ggsave("calibrations_EPLOS_discrimination.pdf", arcalibsplots, width=8, height=8, scale=1.5)

###########################################################################
###########################################################################
###########################################################################
#examine expected cumulative percent of scores in distrubtion for each outcome
###########################################################################
###########################################################################
###########################################################################

#develop a point scoring vector for each outcome category based on selected features' points only (7 in total)

all.equal(dimnames(xmatall)[[2]], dimnames(test_xmatall)[[2]])
ncol(xmatall)
ncol(test_xmatall)

fulldat = rbind.data.frame(xmatall, test_xmatall)
ncol(fulldat)
length(dimnames(fulldat)[[2]])

ss0points = subset(ss0, select=c(feature, sullivan_points))
ss1points = subset(ss1, select=c(feature, sullivan_points))
ss2points = subset(ss2, select=c(feature, sullivan_points))
ss3points = subset(ss3, select=c(feature, sullivan_points))
ss4points = subset(ss4, select=c(feature, sullivan_points))
ss5points = subset(ss5, select=c(feature, sullivan_points))
ss6points = subset(ss6, select=c(feature, sullivan_points))

featlistdat = data.frame(feature = dimnames(fulldat)[[2]])

fullss0points = merge(featlistdat, ss0points, by=c("feature"), all.x=TRUE)
fullss0points[is.na(fullss0points)] = 0

fullss1points = merge(featlistdat, ss1points, by=c("feature"), all.x=TRUE)
fullss1points[is.na(fullss1points)] = 0

fullss2points = merge(featlistdat, ss2points, by=c("feature"), all.x=TRUE)
fullss2points[is.na(fullss2points)] = 0

fullss3points = merge(featlistdat, ss3points, by=c("feature"), all.x=TRUE)
fullss3points[is.na(fullss3points)] = 0

fullss4points = merge(featlistdat, ss4points, by=c("feature"), all.x=TRUE)
fullss4points[is.na(fullss4points)] = 0

fullss5points = merge(featlistdat, ss5points, by=c("feature"), all.x=TRUE)
fullss5points[is.na(fullss5points)] = 0

fullss6points = merge(featlistdat, ss6points, by=c("feature"), all.x=TRUE)
fullss6points[is.na(fullss6points)] = 0

fullsullpoints0 = fullss0points$sullivan_points


phisscore0 = as.matrix(fulldat) %*% fullss0points$sullivan_points
phisscore1 = as.matrix(fulldat) %*% fullss1points$sullivan_points
phisscore2 = as.matrix(fulldat) %*% fullss2points$sullivan_points
phisscore3 = as.matrix(fulldat) %*% fullss3points$sullivan_points
phisscore4 = as.matrix(fulldat) %*% fullss4points$sullivan_points
phisscore5 = as.matrix(fulldat) %*% fullss5points$sullivan_points
phisscore6 = as.matrix(fulldat) %*% fullss6points$sullivan_points

phis_sull_scores = cbind.data.frame(phisscore0, phisscore1, phisscore2, phisscore3, phisscore4, phisscore5, phisscore6)
# phis_sull_scores = cbind.data.frame(phisscore0, phisscore1, phisscore2, phisscore3, phisscore4, phisscore5, phisscore6) %>%
#   mutate_if(is.numeric, as.factor)
summary(phis_sull_scores)
psych::describe(phis_sull_scores)


apply(phis_sull_scores, 2, function(x) quantile(x, probs = seq(.1, .9, value=.1)))

# cumulative_percent_scoring = function(VECTOR, DESCRIPTION_TEXT){
#   fdat = data.frame(table(VECTOR))
#   fdat$total = sum(fdat$Freq)
#   fdat$ptotal = fdat$Freq/fdat$total
#   fdat$pcum = cumsum(fdat$ptotal)
#   fdat$Description = rep(DESCRIPTION_TEXT, times=nrow(fdat))
#   return(fdat)
# }
# 
# cumscorelist = list()
# for(i in 1:ncol(phis_sull_scores)){
#   cumscorelist[[i]] = cumulative_percent_scoring(phis_sull_scores[,i], dimnames(phis_sull_scores)[[2]][i])
# }
# 
# cumscoredat = do.call("rbind.data.frame", cumscorelist)
# cumscoredat$Outcome = factor(as.character(mapvalues(cumscoredat$Description,
#                                                              from=dimnames(phis_sull_scores)[[2]],
#                                                              to=c("a. <1 Month", 
#                                                                   "b. 1 to 2 Months", 
#                                                                   "c. 2 to 3 Months", 
#                                                                   "d. 3 to 4 Months", 
#                                                                   "e. 4 to 5 Months", 
#                                                                   "f. 5 to 6 Months", 
#                                                                   "g. 6+ Months"))))
# cumscoredat$score = as.numeric(as.character(cumscoredat$VECTOR))
# head(cumscoredat)


percentile_rank = function(VECTOR, ROUNDING_DIGITS=2){
  round(trunc(rank(VECTOR, ties="first"))/length(VECTOR), digits=ROUNDING_DIGITS)
}

sull_prcntl_ranks = apply(phis_sull_scores, 2, function(x) percentile_rank(x))

sullrnkwide = cbind.data.frame(phis_sull_scores, sull_prcntl_ranks)
colnames(sullrnkwide) = c(paste0("score", 0:6), paste0("rank", 0:6))
head(sullrnkwide)
sullrnkwide$rowid = 1:nrow(sullrnkwide)

sullrnklong_scores = melt(sullrnkwide,
                   id.vars="rowid",
                   variable.name = "score",
                   measure.vars = c("score0", "score1", "score2", "score3", "score4", "score5", "score6"))
#note that returns following warning since seems to be last variables that excluded:
#"Warning message:
#  attributes are not identical across measure variables; they will be dropped "

sullrnklong_ranks = melt(sullrnkwide,
                          id.vars="rowid",
                          variable.name = "rank",
                          measure.vars = c("rank0", "rank1", "rank2", "rank3", "rank4", "rank5", "rank6"))

sullbind = cbind.data.frame(sullrnklong_scores, sullrnklong_ranks)
colnames(sullbind) = c("rowid1", "outcome1", "score", "rowid2", "outcome2", "prcntl_rank")
all.equal(sullbind[,1], sullbind[,4])
#sullbind$score = factor(sullbind$score)

sullbind$Outcome = factor(as.character(mapvalues(sullbind$outcome2,
                                                             from=c("rank0", "rank1", "rank2", "rank3", "rank4", "rank5", "rank6"),
                                                             to=c("<1 Month",
                                                                  "1 to 2 Months",
                                                                  "2 to 3 Months",
                                                                  "3 to 4 Months",
                                                                  "4 to 5 Months",
                                                                  "5 to 6 Months",
                                                                  "6+ Months"))))


head(sullbind)
summary(sullbind)

distsulls = unique(subset(sullbind, select=c(Outcome, score, prcntl_rank))) %>%
  filter(Outcome!="<1 Month") %>%
  arrange(Outcome, score)
head(distsulls)
str(distsulls)

ggplot(distsulls) +
  geom_smooth(aes(x=score, y=prcntl_rank, colour=Outcome), method = "gam", formula = y ~ s(x, k=12), se = FALSE, size=1) +
  geom_point(aes(x=score, y=prcntl_rank, colour=Outcome), size=.5) +
  #scale_colour_gradient2(low="navy", mid="grey90", high="firebrick", name="Cumulative Proportion") +
  #facet_wrap(~Outcome, nrow=3, scales="free_x") +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,0)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,0)) +
  #scale_x_continuous(limits=c(0, 12), breaks = seq(0, 12, by=1)) +
  scale_y_continuous(limits=c(0, 1), breaks = seq(0, 1, by=.1)) +
  ggtitle("") +
  ylab("Percentile Rank") +
  xlab("Score") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=8),
        axis.text.y=element_text(size=10),
        #panel.grid.major=element_line(colour="grey90", size=.2),
        #panel.grid.major = element_blank(),
        #panel.grid.minor = element_blank(),
        legend.position = "bottom")


##############################################################
##############################################################
##############################################################
#SAVE IMAGE
##############################################################
##############################################################
##############################################################

#save.image("~/MCHS/ELOS/PHIS/keras_ann_full_phis_eplos_2016_2017_cohortV1.RData")


