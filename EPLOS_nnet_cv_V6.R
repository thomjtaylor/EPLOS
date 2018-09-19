

library(NeuralNetTools)
library(nnet)
library(plyr)
library(dplyr)
library(glmnet)
library(survey)
library(ggplot2)
library(caret)
library(pROC)
library(xgboost)

################################################
#note:  script to prepare this is /home/thomas.taylor/Totapally_PICU_KID_EPLOS/R/EPLOS_nnet_cv_V6.R
################################################

rm(list=ls())

setwd("/RPROJECTS/Totapally_PICU_KID_EPLOS/output")

######################################################
######################################################
######################################################
#LOAD EXISTING DATA IMAGE if desired
######################################################
######################################################
######################################################
#load("/DATA/Totapally_PICU_KID_EPLOS/EPLOS_nnet_cv_V1.RData")


#################################################
#select FOLDS to use in modeling
#################################################

global_k_folds = 5


#######################################################
#######################################################
#######################################################
#load data
#######################################################
#######################################################
#######################################################

#NOTE:  the following script (in SVRSCS01 research server) was used to create these cohort analytical files:
#/home/thomas.taylor/Totapally_PICU_KID_EPLOS/R/EPLOS_patient_most_frequent_dx_px_label_identification.R

#full = read.csv("C:/Users/Owner/Documents/MCHS/ELOS/cohorts/neonate_eplos_cohort_all_inidividual_predictors.csv", header=TRUE)[,-1]
#grp = read.csv("C:/Users/Owner/Documents/MCHS/ELOS/cohorts/neonate_eplos_cohort_categorized_predictors.csv", header=TRUE)[,-1]
#summary(full)

full = read.csv("/DATA/Totapally_PICU_KID_EPLOS/neonate_eplos_cohort_all_inidividual_predictors.csv", header=TRUE)[,-1]
grp = read.csv("/DATA/Totapally_PICU_KID_EPLOS/neonate_eplos_cohort_categorized_predictors.csv", header=TRUE)[,-1]
summary(full)

#see creation of clinically informed samples below, right before undersampling k-fold cross validation creation


######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
#setup for predictions with range of classifiers
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################

######################################################
#remove variables not to be included
######################################################
full = subset(full, select=-c(kid_stratum, discwt, los))
grp = subset(grp, select=-c(kid_stratum, discwt, los))


binary_class_imbalance_undersampling_k_fold_cv = function(DATA,
                                                          VARIABLE_TO_SEPARATE_STRING,
                                                          MINORITY_CLASS,
                                                          MINORITY_CLASS_PROPORTION_TO_SAMPLE=1.0,
                                                          K_FOLDS){
  DATA$target = DATA[,grep(VARIABLE_TO_SEPARATE_STRING, names(DATA))]
  minority_class_dat_all = DATA[DATA$target==MINORITY_CLASS,]
  minority_class_dat = minority_class_dat_all[sample(nrow(minority_class_dat_all), ceiling(nrow(minority_class_dat_all)*MINORITY_CLASS_PROPORTION_TO_SAMPLE), replace = FALSE),]
  underdat = DATA[DATA$target!=MINORITY_CLASS,]
  majority_samples_to_create = ceiling(nrow(underdat)/nrow(minority_class_dat))  #make sure that undersampling of majority class is of equal (50/50 split) sample size of minority class by multiply with minority class proportion to sample:  makes the output dataset a 50/50 split that can then be adjusted with LR nomograms for posterior predicted probabilities
  underdat$undersamp_index = sample(1:majority_samples_to_create, nrow(underdat), replace=TRUE)
  unique_majority_class_samples = length(unique(underdat$undersamp_index))
  minority_class_dat$undersamp_index = 0
  minority_class_dat$K_FOLDSindex = sample(1:K_FOLDS, nrow(minority_class_dat), replace=TRUE)
  under_sampling_set = list()
  for(i in 1:unique_majority_class_samples){
    sampunderdat = underdat[underdat$undersamp_index==i,]
    sampunderdat$K_FOLDSindex = sample(1:K_FOLDS, nrow(sampunderdat), replace=TRUE)
    sampdat = rbind.data.frame(minority_class_dat, sampunderdat)
    sampdat1 = subset(sampdat[,-grep(paste0("^", VARIABLE_TO_SEPARATE_STRING, "$"), names(sampdat))], select=-c(undersamp_index, target, K_FOLDSindex))
    #ensure that target is in the first ordinal position of data of interest for later ease of manipulation
    target = sampdat[,grep(VARIABLE_TO_SEPARATE_STRING, names(sampdat))]
    K_FOLDSindex = subset(sampdat, select=c(K_FOLDSindex))
    under_sampling_dat_prep = cbind.data.frame(K_FOLDSindex = K_FOLDSindex, target=target, sampdat1)
    mmsamp_train = list()
    mmsamp_test = list()
    for(k in 1:K_FOLDS){
      mmsamp_train[[k]] = under_sampling_dat_prep[under_sampling_dat_prep$K_FOLDSindex!=k,-c(1)]
      mmsamp_test[[k]] = under_sampling_dat_prep[under_sampling_dat_prep$K_FOLDSindex==k,-c(1)]
    }
    under_sampling_set[[i]] = list(train = mmsamp_train, test=mmsamp_test)
  }
  return(under_sampling_set)
}



#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#NEURAL NETWORK MODELS ESTIMATION
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################

#################################################
#################################################
#################################################
#formula to feed in to neural net models
#################################################
#################################################
#################################################

nnet_formula = as.formula(class.ind(target)~. - synthetic_recnum)
logit_formula = as.formula(target ~ . -1 - synthetic_recnum)

#################################################
#################################################
#################################################
#tuning neuralnet (single hidden layer model)
#################################################
#################################################
#################################################

table(full$elos)
prop.table(table(full$elos))
nrow(full[full$elos==0,])
nrow(unique(subset(full, elos==0, select=-c(recnum))))
#based on unique patient profiles in the non EPLOS, group, create a tuning sample dataframe
tunesamp1 = subset(full, elos==1, select=-c(recnum))
#tunesamp0 = full[sample(nrow(full[full$elos==0,]), 9000),]
tunesamp0 = unique(subset(full, elos==0, select=-c(recnum)))
tunesamp = rbind.data.frame(tunesamp1, tunesamp0)
tunesamp$synthetic_recnum = 1:nrow(tunesamp)
nrow(tunesamp)

table(tunesamp$elos)
prop.table(table(tunesamp$elos))

sims1 = binary_class_imbalance_undersampling_k_fold_cv(DATA=tunesamp, VARIABLE_TO_SEPARATE_STRING = "elos", MINORITY_CLASS = 1,
                                                       MINORITY_CLASS_PROPORTION_TO_SAMPLE=.5, K_FOLDS=global_k_folds)

#sims = sims1[1:250]
sims = sims1
rm(sims1)
length(sims)


#################################################
#################################################
#################################################
#find optimal decay through CV on full dataset
#after experimenting with multiple runs, most parsimonious seems to be 6 hidden layers with 0 decay
#################################################
#################################################
#################################################

# tunegrid_nnet = expand.grid(decay_regularization_range = c(.01, .1, .5, 1, 5),
#                             hidden_layer_range = c(4, 6, 8, 10))
# tunegrid_nnet$index = 1:nrow(tunegrid_nnet)
# 
# length(sims)*global_k_folds*nrow(tunegrid_nnet)
# 
# 
# starttime = proc.time()
# nnmodi = list()
# nnpredi = list()
# for(i in 1:length(sims)){
#   train_folds = length(sims)
#   #test_folds = length(sims) #same as train_folds
#   #for each undersampled dataset, run k-fold cross-validation training and testing
#   nnmodj = list()
#   nnpredj = list()
#   for(j in 1:global_k_folds){
#     nnmodk_n = list()
#     nnmodk_nunits = list()
#     nnmodk_nconn = list()
#     nnmodk_nsunits = list()
#     nnmodk_decay = list()
#     nnmodk_wts = list()
#     nnmodk_value = list()
#     #nnmodk_fitted.values = list() #toggled off to keep object size down for large databases
#     #nnmodk_residuals = list() #toggled off to keep object size down for large databases
#     nnmodk_convergence = list()
#     nnmodk_coefnames = list()
#     nnpredk = list()
#     for(k in 1:nrow(tunegrid_nnet)){
#       nnmodk = nnet(nnet_formula, data=sims[[i]]$train[[j]], decay=tunegrid_nnet$decay_regularization_range[k], size=tunegrid_nnet$hidden_layer_range[k], softmax=TRUE, maxit=100)
#       nnmodk_n[[k]] = nnmodk$n
#       nnmodk_nunits[[k]] = nnmodk$nunits
#       nnmodk_nconn[[k]] = nnmodk$nconn
#       nnmodk_nsunits[[k]] = nnmodk$nsunits
#       nnmodk_decay[[k]] = nnmodk$decay
#       nnmodk_wts[[k]] = nnmodk$wts
#       nnmodk_value[[k]] = nnmodk$value
#       nnmodk_convergence[[k]] = ifelse(nnmodk$convergence==1, "max iterations reached", "stopped before max iterations reached")
#       nnmodk_coefnames[[k]] = nnmodk$coefnames
#       nnpredk[[k]] = data.frame(predict(nnmodk, newdata=sims[[i]]$test[[j]], type="raw"))
#       nnpredk[[k]]$target = sims[[i]]$test[[j]]$target
#       nnpredk[[k]]$synthetic_recnum = sims[[i]]$test[[j]]$synthetic_recnum 
#       nnpredk[[k]]$decay_parm = tunegrid_nnet$decay_regularization_range[k]
#       nnpredk[[k]]$hidden_layers = tunegrid_nnet$hidden_layer_range[k]
#       nnpredk[[k]]$fold = j
#       nnpredk[[k]]$sample = i
#     }
#     nnmodj[[j]] = list(nstruct = nnmodk_n,
#                        nunits=nnmodk_nunits,
#                        value = nnmodk_value,
#                        convergence = nnmodk_convergence,
#                        wts = nnmodk_wts,
#                        coefnames = nnmodk_coefnames)
#     nnpredj[[j]] = do.call("rbind.data.frame", nnpredk)
#   }
#   nnmodi[[i]] = list(nnmodj)
#   nnpredi[[i]] = list(nnpredj)
# }
# stoptime = proc.time()
# hoursruntime = (stoptime-starttime)/3600
# hoursruntime
# 
# nnpredsall = list()
# for(i in 1:length(nnpredi)){
#   nnpredsall[[i]] = nnpredi[[i]][[1]][[1]]
# }
# 
# tune_nnpredsalldat = do.call("rbind.data.frame", nnpredsall)
# tune_nnpredsalldat$lnX0 = log(tune_nnpredsalldat$X0)
# tune_nnpredsalldat$lnX1 = log(tune_nnpredsalldat$X1)
# 
# tune_nnpredsum = tune_nnpredsalldat %>%
#   group_by(decay_parm, hidden_layers, target) %>%
#   summarize(n = n(),
#             meanlnX0 = mean(lnX0, na.rm=TRUE),
#             meanlnX1 = mean(lnX1, na.rm=TRUE),
#             sdlnX0 = sd(lnX0, na.rm=TRUE),
#             sdlnX1 = sd(lnX1, na.rm=TRUE))
# tune_nnpredsum$X0 = round(exp(tune_nnpredsum$meanlnX0), digits=2)
# tune_nnpredsum$X0LL = round(exp(tune_nnpredsum$meanlnX0 - 1.96*(tune_nnpredsum$sdlnX0/sqrt(360))), digits=2) #n denominator for SE seems most appropriately to be number of under-samplings*number of folds
# tune_nnpredsum$X0UL = round(exp(tune_nnpredsum$meanlnX0 + 1.96*(tune_nnpredsum$sdlnX0/sqrt(360))), digits=2) #n denominator for SE seems most appropriately to be number of under-samplings*number of folds
# tune_nnpredsum$X1 = round(exp(tune_nnpredsum$meanlnX1), digits=2)
# tune_nnpredsum$X1LL = round(exp(tune_nnpredsum$meanlnX1 - 1.96*(tune_nnpredsum$sdlnX1/sqrt(360))), digits=2) #n denominator for SE seems most appropriately to be number of under-samplings*number of folds
# tune_nnpredsum$X1UL = round(exp(tune_nnpredsum$meanlnX1 + 1.96*(tune_nnpredsum$sdlnX1/sqrt(360))), digits=2) #n denominator for SE seems most appropriately to be number of under-samplings*number of folds
# 
# data.frame(tune_nnpredsum %>% arrange(-X1))

######################################################
######################################################
######################################################
#fit best tuned model among tuning to FULL predictor dataset
######################################################
######################################################
######################################################

#model tuning suggests cross validation highest average propbabilities for the outcome (1) of EPLOS, select:
#decay=1.0, hidden node=6

#NOTE!
#"_tps" = "tune paramaters set"

# table(full$elos)
# prop.table(table(full$elos))
# prop.table(table(full$elos))[[2]]

sims_tps = binary_class_imbalance_undersampling_k_fold_cv(DATA=tunesamp, VARIABLE_TO_SEPARATE_STRING = "elos", MINORITY_CLASS = 1, 
                                                          MINORITY_CLASS_PROPORTION_TO_SAMPLE=1.0, K_FOLDS=global_k_folds)

length(sims_tps)*global_k_folds

starttime_tps = proc.time()
nnmodi_tps = list()
nnpredi_tps = list()
for(i in 1:length(sims_tps)){
  #for each undersampled dataset, run k-fold cross-validation training and testing
  nnmodj_tps = list()
  nnpredj_tps = list()
  for(j in 1:global_k_folds){
    nnmodj_tps[[j]] = nnet(nnet_formula, data=sims_tps[[i]]$train[[j]], decay=1.0, size=6, softmax=TRUE, maxit=1000)
    nnpredj_tps[[j]] = data.frame(predict(nnmodj_tps[[j]], newdata=sims_tps[[i]]$test[[j]], type="raw"))
    nnpredj_tps[[j]]$target = sims_tps[[i]]$test[[j]]$target
    nnpredj_tps[[j]]$synthetic_recnum = sims_tps[[i]]$test[[j]]$synthetic_recnum 
    nnpredj_tps[[j]]$fold = j
    nnpredj_tps[[j]]$sample = i
  }
  nnmodi_tps[[i]] = list(nnmodj_tps)
  nnpredi_tps[[i]] = do.call("rbind.data.frame", nnpredj_tps)
  nnpredi_tps[[i]]$discharge_id = dimnames(nnpredi_tps[[i]])[[1]]
}
stoptime_tps = proc.time()
hoursruntime_tps = (stoptime_tps-starttime_tps)/3600
hoursruntime_tps

######################################################
######################################################
######################################################
#fit best tuned model among tuning to GROUPED predictor dataset
######################################################
######################################################
######################################################

#model tuning suggests cross validation highest average propbabilities for the outcome (1) of EPLOS, select:
#decay=1.0, hidden node=6

#NOTE!
#"_tps" = "tune paramaters set"

#######################################################
#use tuning sample, but with grouped indicators
#######################################################

table(grp$elos)
prop.table(table(grp$elos))
nrow(full[grp$elos==0,])
nrow(unique(subset(grp, elos==0, select=-c(recnum))))
#based on unique patient profiles in the non EPLOS, group, create a tuning sample dataframe
grp_tunesamp1 = subset(grp, elos==1, select=-c(recnum))
grp_tunesamp0 = unique(subset(grp, elos==0, select=-c(recnum)))
grp_tunesamp = rbind.data.frame(grp_tunesamp1, grp_tunesamp0)
grp_tunesamp$synthetic_recnum = 1:nrow(grp_tunesamp)
nrow(grp_tunesamp)


grp_sims_tps = binary_class_imbalance_undersampling_k_fold_cv(DATA=grp_tunesamp, VARIABLE_TO_SEPARATE_STRING = "elos", MINORITY_CLASS = 1, 
                                                              MINORITY_CLASS_PROPORTION_TO_SAMPLE=1, K_FOLDS=global_k_folds)
length(grp_sims_tps)*global_k_folds

starttime_grp_tps = proc.time()
nnmodi_grp_tps = list()
nnpredi_grp_tps = list()
for(i in 1:length(grp_sims_tps)){
  train_folds_grp_tps = length(grp_sims_tps)
  #test_folds_grp_tps = length(sims) #same as train_folds
  #for each undersampled dataset, run k-fold cross-validation training and testing
  nnmodj_grp_tps = list()
  nnpredj_grp_tps = list()
  for(j in 1:global_k_folds){
    nnmodj_grp_tps[[j]] = nnet(nnet_formula, data=grp_sims_tps[[i]]$train[[j]], decay=1.0, size=6, softmax=TRUE, maxit=1000)
    nnpredj_grp_tps[[j]] = data.frame(predict(nnmodj_grp_tps[[j]], newdata=grp_sims_tps[[i]]$test[[j]], type="raw"))
    nnpredj_grp_tps[[j]]$target = grp_sims_tps[[i]]$test[[j]]$target
    nnpredj_grp_tps[[j]]$synthetic_recnum = grp_sims_tps[[i]]$test[[j]]$synthetic_recnum 
    nnpredj_grp_tps[[j]]$fold = j
    nnpredj_grp_tps[[j]]$sample = i
  }
  nnmodi_grp_tps[[i]] = list(nnmodj_grp_tps)
  nnpredi_grp_tps[[i]] = do.call("rbind.data.frame", nnpredj_grp_tps) 
  nnpredi_grp_tps[[i]]$discharge_id = dimnames(nnpredi_grp_tps[[i]])[[1]]
}
stoptime_grp_tps = proc.time()
hoursruntime_grp_tps = (stoptime_grp_tps-starttime_grp_tps)/3600
hoursruntime_grp_tps



#######################################################
#clinical indicators selected (V1)
#######################################################

#NOTE: using same number of samples as grouped model (which is the smaller tuning sample between grouped and full,)

######################################################
#create clinically informed grouping based on most influential features & Dr. Totapally's selection (4/29/2018 email)
######################################################

clin1 = subset(tunesamp, select=c("synthetic_recnum", "elos", 
                              "X3885...Occl.thoracic.vess.NEC",
                              "X74783...Persist.fetal.circulat",
                              "X7483...Laryngotrach.anomaly.NEC",
                              "X3965...ECMO",
                              "X5849...Acute.kidney.failure.NOS",
                              "X9983...Phototherapy.NEC",
                              "X2762...Acidosis",
                              "X4610...Colostomy.NOS",
                              "X2761...Hyposmolality",
                              "X4562...Part.S.intest.resect.NEC",
                              "X77182...NB.urinary.tract.infect",
                              "X77750...Necr.enterocol.NB.NOS",
                              "X76502...Extreme.immatur.500.749g",
                              "X76503...Extreme.immatur.750.999g",
                              "X76522...24.weeks.gestation",
                              "X76523...25.26.weeks.gestation"
))
summary(clin1)
head(clin1)
names(clin1)

table(clin1$elos)
prop.table(table(clin1$elos))
nrow(full[clin1$elos==0,])
nrow(unique(subset(clin1, elos==0)))
clin1_tunesamp1 = subset(clin1, elos==1)
clin1_tunesamp0 = unique(subset(clin1, elos==0))
clin1_tunesamp = rbind.data.frame(clin1_tunesamp1, clin1_tunesamp0)
#NOTE: using full sample, since feature set is so small so estimation may be quicker
#based on unique patient profiles in the non EPLOS, group, create a tuning sample dataframe
# clin1_tunesamp = clin1
clin1_tunesamp$synthetic_recnum = 1:nrow(clin1_tunesamp)
nrow(clin1_tunesamp)


clin1_sims_tps = binary_class_imbalance_undersampling_k_fold_cv(DATA=clin1, VARIABLE_TO_SEPARATE_STRING = "elos", MINORITY_CLASS = 1, 
                                                                MINORITY_CLASS_PROPORTION_TO_SAMPLE=1, K_FOLDS=global_k_folds)
length(clin1_sims_tps)*global_k_folds

starttime_clin1_tps = proc.time()
nnmodi_clin1_tps = list()
nnpredi_clin1_tps = list()
for(i in 1:length(clin1_sims_tps)){
  train_folds_clin1_tps = length(clin1_sims_tps)
  #test_folds_clin1_tps = length(sims) #same as train_folds
  #for each undersampled dataset, run k-fold cross-validation training and testing
  nnmodj_clin1_tps = list()
  nnpredj_clin1_tps = list()
  for(j in 1:global_k_folds){
    nnmodj_clin1_tps[[j]] = nnet(nnet_formula, data=clin1_sims_tps[[i]]$train[[j]], decay=1.0, size=6, softmax=TRUE, maxit=1000)
    nnpredj_clin1_tps[[j]] = data.frame(predict(nnmodj_clin1_tps[[j]], newdata=clin1_sims_tps[[i]]$test[[j]], type="raw"))
    nnpredj_clin1_tps[[j]]$target = clin1_sims_tps[[i]]$test[[j]]$target
    nnpredj_clin1_tps[[j]]$synthetic_recnum = clin1_sims_tps[[i]]$test[[j]]$synthetic_recnum 
    nnpredj_clin1_tps[[j]]$fold = j
    nnpredj_clin1_tps[[j]]$sample = i
  }
  nnmodi_clin1_tps[[i]] = list(nnmodj_clin1_tps)
  nnpredi_clin1_tps[[i]] = do.call("rbind.data.frame", nnpredj_clin1_tps) 
  nnpredi_clin1_tps[[i]]$discharge_id = dimnames(nnpredi_clin1_tps[[i]])[[1]]
}
stoptime_clin1_tps = proc.time()
hoursruntime_clin1_tps = (stoptime_clin1_tps-starttime_clin1_tps)/3600
hoursruntime_clin1_tps


######################################################
######################################################
######################################################
#Neural Network Variable importance assessment across undersampling and k-fold cross validations
######################################################
######################################################
######################################################

olden(nnmodi_tps[[1]][[1]][[2]], out_var="1")
olden(nnmodi_grp_tps[[1]][[1]][[2]], out_var="1")

length(sims_tps); length(grp_sims_tps); length(clin1_sims_tps);

var_import_full_fold_0 = list()
var_import_full_fold_1 = list()
var_import_full_fold = list()
var_import_grp_fold_0 = list()
var_import_grp_fold_1 = list()
var_import_grp_fold = list()
var_import_full_und = list()
var_import_grp_und = list()
var_import_clin1_fold_0 = list()
var_import_clin1_fold_1 = list()
var_import_clin1_fold = list()
var_import_clin1_und = list()

for(i in 1:length(clin1_sims_tps)){
  for(j in 1:global_k_folds){
    #full covariate list
    var_import_full_fold_0[[j]] = data.frame(olden(nnmodi_tps[[i]][[1]][[j]], bar_plot=FALSE, out_var="0"))
    var_import_full_fold_0[[j]]$coefnames = dimnames(olden(nnmodi_tps[[i]][[1]][[j]], bar_plot=FALSE, out_var="0"))[[1]]
    var_import_full_fold_0[[j]]$outcome = 0
    var_import_full_fold_0[[j]]$fold = j
    var_import_full_fold_0[[j]]$sample = i
    var_import_full_fold_1[[j]] = data.frame(olden(nnmodi_tps[[i]][[1]][[j]], bar_plot=FALSE, out_var="1"))
    var_import_full_fold_1[[j]]$coefnames = dimnames(olden(nnmodi_tps[[i]][[1]][[j]], bar_plot=FALSE, out_var="1"))[[1]]
    var_import_full_fold_1[[j]]$outcome = 1
    var_import_full_fold_1[[j]]$fold = j
    var_import_full_fold_1[[j]]$sample = i
    var_import_full_fold[[j]] = rbind.data.frame(var_import_full_fold_0[[j]], var_import_full_fold_1[[j]])
    #group covariate list
    var_import_grp_fold_0[[j]] = data.frame(olden(nnmodi_grp_tps[[i]][[1]][[j]], bar_plot=FALSE, out_var="0"))
    var_import_grp_fold_0[[j]]$coefnames = dimnames(olden(nnmodi_grp_tps[[i]][[1]][[j]], bar_plot=FALSE, out_var="0"))[[1]]
    var_import_grp_fold_0[[j]]$outcome = 0
    var_import_grp_fold_0[[j]]$fold = j
    var_import_grp_fold_0[[j]]$sample = i
    var_import_grp_fold_1[[j]] = data.frame(olden(nnmodi_grp_tps[[i]][[1]][[j]], bar_plot=FALSE, out_var="1"))
    var_import_grp_fold_1[[j]]$coefnames = dimnames(olden(nnmodi_grp_tps[[i]][[1]][[j]], bar_plot=FALSE, out_var="1"))[[1]]
    var_import_grp_fold_1[[j]]$outcome = 1
    var_import_grp_fold_1[[j]]$fold = j
    var_import_grp_fold_1[[j]]$sample = i
    var_import_grp_fold[[j]] = rbind.data.frame(var_import_grp_fold_0[[j]], var_import_grp_fold_1[[j]])
    #clinical covariate list
    var_import_clin1_fold_0[[j]] = data.frame(olden(nnmodi_clin1_tps[[i]][[1]][[j]], bar_plot=FALSE, out_var="0"))
    var_import_clin1_fold_0[[j]]$coefnames = dimnames(olden(nnmodi_clin1_tps[[i]][[1]][[j]], bar_plot=FALSE, out_var="0"))[[1]]
    var_import_clin1_fold_0[[j]]$outcome = 0
    var_import_clin1_fold_0[[j]]$fold = j
    var_import_clin1_fold_0[[j]]$sample = i
    var_import_clin1_fold_1[[j]] = data.frame(olden(nnmodi_clin1_tps[[i]][[1]][[j]], bar_plot=FALSE, out_var="1"))
    var_import_clin1_fold_1[[j]]$coefnames = dimnames(olden(nnmodi_clin1_tps[[i]][[1]][[j]], bar_plot=FALSE, out_var="1"))[[1]]
    var_import_clin1_fold_1[[j]]$outcome = 1
    var_import_clin1_fold_1[[j]]$fold = j
    var_import_clin1_fold_1[[j]]$sample = i
    var_import_clin1_fold[[j]] = rbind.data.frame(var_import_clin1_fold_0[[j]], var_import_clin1_fold_1[[j]])
  }
  var_import_full_und[[i]] = do.call("rbind.data.frame", var_import_full_fold)
  var_import_grp_und[[i]] = do.call("rbind.data.frame", var_import_grp_fold)
  var_import_clin1_und[[i]] = do.call("rbind.data.frame", var_import_clin1_fold)
}

var_import_full = do.call("rbind.data.frame", var_import_full_und)
var_import_grp = do.call("rbind.data.frame", var_import_grp_und)
var_import_clin1 = do.call("rbind.data.frame", var_import_clin1_und)

var_import_full_des = svydesign(id=~sample + fold, weights=~1, data=var_import_full[var_import_full$outcome==1,]) #since all weights are reversed between 0 and 1 in binary case
var_import_grp_des = svydesign(id=~sample + fold, weights=~1, data=var_import_grp[var_import_grp$outcome==1,]) #since all weights are reversed between 0 and 1 in binary case
var_import_clin1_des = svydesign(id=~sample + fold, weights=~1, data=var_import_clin1[var_import_clin1$outcome==1,]) #since all weights are reversed between 0 and 1 in binary case


full_var_import_means = cbind.data.frame(svyby(~importance, ~outcome + coefnames, design=var_import_full_des, svymean),
                                         confint(svyby(~importance, ~outcome + coefnames, design=var_import_full_des, svymean)))
full_var_import_means$Variable_Set = "Full List"

grp_var_import_means = cbind.data.frame(svyby(~importance, ~outcome + coefnames, design=var_import_grp_des, svymean),
                                         confint(svyby(~importance, ~outcome + coefnames, design=var_import_grp_des, svymean)))
grp_var_import_means$Variable_Set = "Grouped List"

clin1_var_import_means = cbind.data.frame(svyby(~importance, ~outcome + coefnames, design=var_import_clin1_des, svymean),
                                        confint(svyby(~importance, ~outcome + coefnames, design=var_import_clin1_des, svymean)))
clin1_var_import_means$Variable_Set = "Clinicaly Guided List"


var_import_means = rbind.data.frame(full_var_import_means, grp_var_import_means, clin1_var_import_means)
var_import_means$rowid = 1:nrow(var_import_means)
colnames(var_import_means) = c("outcome", "coefnames", "importance", "se", "LL95", "UL95", "Variable_Set", "rowid")
head(var_import_means)

var_import_means$new_coefnames = gsub("[...]", " ", sub("X", "", var_import_means$coefnames))

dev.off()
var_import_means_plot = ggplot(var_import_means) +
  geom_hline(yintercept=0, colour="black", size=.3) +
  geom_errorbar(aes(x=reorder(new_coefnames, -importance), ymin=LL95, ymax=UL95), width=0, size=1) +
  geom_point(data=var_import_means, aes(x=reorder(new_coefnames, -importance), y=importance, colour=importance), size=3) +
  coord_flip() + 
  scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  facet_wrap(~Variable_Set, nrow=1) +
  ylab("Mean Variable Importance") +
  xlab("Variable") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.minor = element_blank(),
        legend.position = "bottom")
var_import_means_plot
ggsave("/RPROJECTS/Totapally_PICU_KID_EPLOS/variable_importance_means_initial_neural_net_6_node_estimates.pdf", var_import_means_plot, width=11, height=8, scale=1.4)

######################################################
######################################################
######################################################
#weights averaging over undersampling draws and k-fold cross-validation
######################################################
######################################################
######################################################

mclfold = list()
mcldat = list()
wtsfold = list()
wtsdat = list()

for(i in 1:length(sims_tps)){
  for(j in 1:global_k_folds){
    #maximum conditional likelihood estimates
    mclfold[[j]] = data.frame(mcl = nnmodi_tps[[i]][[1]][[j]]$value)
    mclfold[[j]]$fold = j
    #neuron weights
    wtsfold[[j]] = data.frame(wts = nnmodi_tps[[i]][[1]][[j]]$wts)
    wtsfold[[j]]$row = 1:nrow(wtsfold[[j]])
    wtsfold[[j]]$fold = j
  }  
  mcldat[[i]] = do.call("rbind.data.frame", mclfold)
  mcldat[[i]]$sample = i
  wtsdat[[i]] = do.call("rbind.data.frame", wtsfold)
  wtsdat[[i]]$sample = i
}

mclvaldat = do.call("rbind.data.frame", mcldat)
nrow(mclvaldat)
wtslongdat = do.call("rbind.data.frame", wtsdat)
nrow(wtslongdat)
head(wtslongdat)
summary(wtslongdat)

wtsagg = psych::describeBy(wtslongdat$wts, group=wtslongdat$row, mat=TRUE, digits=3)

#?plotnet
fullstruct = nnmodi_tps[[i]][[1]][[j]]$n
plotnet(mod_in=wtsagg$mean, struct=fullstruct)

######################################################
######################################################
######################################################
#Nueral Network K-fold-cross-validated under-sampled predictions
######################################################
######################################################
######################################################

nnpred_full_tps = do.call("rbind.data.frame", nnpredi_tps)
nnpred_grp_tps = do.call("rbind.data.frame", nnpredi_grp_tps)
nnpred_clin1_tps = do.call("rbind.data.frame", nnpredi_clin1_tps)


head(nnpred_full_tps)
nrow(nnpred_full_tps)
nnpred_full_tps_agg = nnpred_full_tps %>% 
  group_by(synthetic_recnum, target) %>%
  summarize(n = n(),
            mX1 = mean(X1, na.rm=TRUE),
            mX0 = mean(X0, na.rm=TRUE),
            sdX1 = sd(X1, na.rm=TRUE),
            sdX0 = sd(X0, na.rm=TRUE))
nnpred_full_tps_agg
nnpred_full_tps_agg$Variable_List = "Full"

head(nnpred_grp_tps)
nrow(nnpred_grp_tps)
nnpred_grp_tps_agg = nnpred_grp_tps %>% 
  group_by(synthetic_recnum, target) %>%
  summarize(n = n(),
            mX1 = mean(X1, na.rm=TRUE),
            mX0 = mean(X0, na.rm=TRUE),
            sdX1 = sd(X1, na.rm=TRUE),
            sdX0 = sd(X0, na.rm=TRUE))
nnpred_grp_tps_agg
nnpred_grp_tps_agg$Variable_List = "Grouped"


head(nnpred_clin1_tps)
nrow(nnpred_clin1_tps)
nnpred_clin1_tps_agg = nnpred_clin1_tps %>% 
  group_by(synthetic_recnum, target) %>%
  summarize(n = n(),
            mX1 = mean(X1, na.rm=TRUE),
            mX0 = mean(X0, na.rm=TRUE),
            sdX1 = sd(X1, na.rm=TRUE),
            sdX0 = sd(X0, na.rm=TRUE))
nnpred_clin1_tps_agg
nnpred_clin1_tps_agg$Variable_List = "Clinically Guided"


######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
#SIMPLE LOGISTIC REGRESSION PREDICTIONS
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################

starttime_tps = proc.time()
logit_modi_tps = list()
logit_coefsi_tps = list()
logit_predi_tps = list()
for(i in 1:length(sims_tps)){
  #for each undersampled dataset, run k-fold cross-validation training and testing
  logit_modj_tps = list()
  logit_coefsj_tps = list()
  logit_predj_tps = list()
  for(j in 1:global_k_folds){
    logit_modj_tps[[j]] = glm(logit_formula, data=sims_tps[[i]]$train[[j]], family="binomial")
    #coefficients
    logit_coefsj_tps[[j]] = data.frame(Feature = dimnames(data.frame(logit_modj_tps[[j]]$coefficients))[1], estimate = as.vector(logit_modj_tps[[j]]$coefficients))
    rownames(logit_coefsj_tps[[j]]) = NULL
    logit_coefsj_tps[[j]]$fold = j
    logit_coefsj_tps[[j]]$sample = i
    #predictions
    logit_predj_tps[[j]] = data.frame(predict(logit_modj_tps[[j]], newdata=sims_tps[[i]]$test[[j]], type="response"))
    logit_predj_tps[[j]]$target = sims_tps[[i]]$test[[j]]$target
    logit_predj_tps[[j]]$synthetic_recnum = sims_tps[[i]]$test[[j]]$synthetic_recnum 
    logit_predj_tps[[j]]$fold = j
    logit_predj_tps[[j]]$sample = i
  }
  logit_modi_tps[[i]] = list(logit_modj_tps)
  logit_coefsi_tps[[i]] = do.call("rbind.data.frame", logit_coefsj_tps)
  logit_predi_tps[[i]] = do.call("rbind.data.frame", logit_predj_tps)
  logit_predi_tps[[i]]$discharge_id = dimnames(logit_predi_tps[[i]])[[1]]
}
stoptime_tps = proc.time()
hoursruntime_tps = (stoptime_tps-starttime_tps)/3600
hoursruntime_tps

starttime_grp_tps = proc.time()
logit_modi_grp_tps = list()
logit_coefsi_grp_tps = list()
logit_predi_grp_tps = list()
for(i in 1:length(grp_sims_tps)){
  train_folds_grp_tps = length(grp_sims_tps)
  #test_folds_grp_tps = length(sims) #same as train_folds
  #for each undersampled dataset, run k-fold cross-validation training and testing
  logit_modj_grp_tps = list()
  logit_coefsj_grp_tps = list()
  logit_predj_grp_tps = list()
  for(j in 1:global_k_folds){
    logit_modj_grp_tps[[j]] = glm(logit_formula, data=grp_sims_tps[[i]]$train[[j]], family="binomial")
    #coefficients
    logit_coefsj_grp_tps[[j]] = data.frame(Feature = dimnames(data.frame(logit_modj_grp_tps[[j]]$coefficients))[1], estimate = as.vector(logit_modj_grp_tps[[j]]$coefficients))
    rownames(logit_coefsj_grp_tps[[j]]) = NULL
    logit_coefsj_grp_tps[[j]]$fold = j
    logit_coefsj_grp_tps[[j]]$sample = i
    #predictions
    logit_predj_grp_tps[[j]] = data.frame(predict(logit_modj_grp_tps[[j]], newdata=grp_sims_tps[[i]]$test[[j]], type="response"))
    logit_predj_grp_tps[[j]]$target = grp_sims_tps[[i]]$test[[j]]$target
    logit_predj_grp_tps[[j]]$synthetic_recnum = grp_sims_tps[[i]]$test[[j]]$synthetic_recnum 
    logit_predj_grp_tps[[j]]$fold = j
    logit_predj_grp_tps[[j]]$sample = i
  }
  logit_modi_grp_tps[[i]] = list(logit_modj_grp_tps)
  logit_coefsi_grp_tps[[i]] = do.call("rbind.data.frame", logit_coefsj_grp_tps)
  logit_predi_grp_tps[[i]] = do.call("rbind.data.frame", logit_predj_grp_tps) 
  logit_predi_grp_tps[[i]]$discharge_id = dimnames(logit_predi_grp_tps[[i]])[[1]]
}
stoptime_grp_tps = proc.time()
hoursruntime_grp_tps = (stoptime_grp_tps-starttime_grp_tps)/3600
hoursruntime_grp_tps

starttime_clin1_tps = proc.time()
logit_modi_clin1_tps = list()
logit_coefsi_clin1_tps = list()
logit_predi_clin1_tps = list()
for(i in 1:length(clin1_sims_tps)){
  train_folds_clin1_tps = length(clin1_sims_tps)
  #test_folds_clin1_tps = length(sims) #same as train_folds
  #for each undersampled dataset, run k-fold cross-validation training and testing
  logit_modj_clin1_tps = list()
  logit_coefsj_clin1_tps = list()
  logit_predj_clin1_tps = list()
  for(j in 1:global_k_folds){
    logit_modj_clin1_tps[[j]] = glm(logit_formula, data=clin1_sims_tps[[i]]$train[[j]], family="binomial")
    #coefficients
    logit_coefsj_clin1_tps[[j]] = data.frame(Feature = dimnames(data.frame(logit_modj_clin1_tps[[j]]$coefficients))[1], estimate = as.vector(logit_modj_clin1_tps[[j]]$coefficients))
    rownames(logit_coefsj_clin1_tps[[j]]) = NULL
    logit_coefsj_clin1_tps[[j]]$fold = j
    logit_coefsj_clin1_tps[[j]]$sample = i
    #predictions
    logit_predj_clin1_tps[[j]] = data.frame(predict(logit_modj_clin1_tps[[j]], newdata=clin1_sims_tps[[i]]$test[[j]], type="response"))
    logit_predj_clin1_tps[[j]]$target = clin1_sims_tps[[i]]$test[[j]]$target
    logit_predj_clin1_tps[[j]]$synthetic_recnum = clin1_sims_tps[[i]]$test[[j]]$synthetic_recnum 
    logit_predj_clin1_tps[[j]]$fold = j
    logit_predj_clin1_tps[[j]]$sample = i
  }
  logit_modi_clin1_tps[[i]] = list(logit_modj_clin1_tps)
  logit_coefsi_clin1_tps[[i]] = do.call("rbind.data.frame", logit_coefsj_clin1_tps)
  logit_predi_clin1_tps[[i]] = do.call("rbind.data.frame", logit_predj_clin1_tps) 
  logit_predi_clin1_tps[[i]]$discharge_id = dimnames(logit_predi_clin1_tps[[i]])[[1]]
}
stoptime_clin1_tps = proc.time()
hoursruntime_clin1_tps = (stoptime_clin1_tps-starttime_clin1_tps)/3600
hoursruntime_clin1_tps



##############################################################
##############################################################
##############################################################
#averaging coefficients across repeated k-fold cross validation (for comparison with other methods' variable importance estimates)
##############################################################
##############################################################
##############################################################

var_coefs_full = do.call("rbind.data.frame", logit_coefsi_tps)
colnames(var_coefs_full) = c("coefnames", "estimate", "fold", "sample")
var_coefs_grp = do.call("rbind.data.frame", logit_coefsi_grp_tps)
colnames(var_coefs_grp) = c("coefnames", "estimate", "fold", "sample")
var_coefs_clin1 = do.call("rbind.data.frame", logit_coefsi_clin1_tps)
colnames(var_coefs_clin1) = c("coefnames", "estimate", "fold", "sample")


var_coefs_full_des = svydesign(id=~sample + fold, weights=~1, data=var_coefs_full)
var_coefs_grp_des = svydesign(id=~sample + fold, weights=~1, data=var_coefs_grp)
var_coefs_clin1_des = svydesign(id=~sample + fold, weights=~1, data=var_coefs_clin1)


full_var_coefs_means = cbind.data.frame(svyby(~estimate, ~coefnames, design=var_coefs_full_des, svymean),
                                         confint(svyby(~estimate, ~coefnames, design=var_coefs_full_des, svymean)))
full_var_coefs_means$Variable_Set = "Full List"

grp_var_coefs_means = cbind.data.frame(svyby(~estimate, ~coefnames, design=var_coefs_grp_des, svymean),
                                        confint(svyby(~estimate, ~coefnames, design=var_coefs_grp_des, svymean)))
grp_var_coefs_means$Variable_Set = "Grouped List"

clin1_var_coefs_means = cbind.data.frame(svyby(~estimate, ~coefnames, design=var_coefs_clin1_des, svymean),
                                       confint(svyby(~estimate, ~coefnames, design=var_coefs_clin1_des, svymean)))
clin1_var_coefs_means$Variable_Set = "Clinically Guided List"


var_coefs_means = rbind.data.frame(full_var_coefs_means, grp_var_coefs_means, clin1_var_coefs_means)
var_coefs_means$rowid = 1:nrow(var_coefs_means)
colnames(var_coefs_means) = c("Coefficient", "beta", "se", "LL95", "UL95", "Variable_Set", "rowid")
head(var_coefs_means)

var_coefs_means$new_coefnames = gsub("[...]", " ", sub("X", "", var_coefs_means$Coefficient))

dev.off()
var_coefs_means_plot = ggplot(var_coefs_means) +
  geom_hline(yintercept=0, colour="black", size=.3) +
  geom_errorbar(aes(x=reorder(new_coefnames, -beta), ymin=LL95, ymax=UL95), width=0, size=1) +
  geom_point(data=var_coefs_means, aes(x=reorder(new_coefnames, -beta), y=beta, colour=beta), size=3) +
  coord_flip() +
  #scale_y_continuous(limits=c(-30, 30)) +
  scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Logit Magnitude") +
  facet_wrap(~Variable_Set, nrow=1) +
  ylab("Mean Variable Importance") +
  xlab("Variable") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.minor = element_blank(),
        legend.position = "bottom")
var_coefs_means_plot
ggsave("/RPROJECTS/Totapally_PICU_KID_EPLOS/variableimportance_from_rep_k_fold_logistic_glms.pdf", var_coefs_means_plot, width=11, height=8, scale=1.4)


######################################################
######################################################
######################################################
#Logistic Regression K-fold-cross-validated under-sampled predictions
######################################################
######################################################
######################################################

logit_pred_full_tps = do.call("rbind.data.frame", logit_predi_tps)
colnames(logit_pred_full_tps) = c("glmprob", "target", "synthetic_recnum", "fold", "sample", "discharge_id")
logit_pred_grp_tps = do.call("rbind.data.frame", logit_predi_grp_tps)
colnames(logit_pred_grp_tps) = c("glmprob", "target", "synthetic_recnum", "fold", "sample", "discharge_id")
logit_pred_clin1_tps = do.call("rbind.data.frame", logit_predi_clin1_tps)
colnames(logit_pred_clin1_tps) = c("glmprob", "target", "synthetic_recnum", "fold", "sample", "discharge_id")



summary(logit_pred_full_tps)
summary(logit_pred_grp_tps)
summary(logit_pred_clin1_tps)

head(logit_pred_full_tps)
nrow(logit_pred_full_tps)
logit_pred_full_tps_agg = logit_pred_full_tps %>% 
  group_by(synthetic_recnum, target) %>%
  summarize(n = n(),
            mX1 = mean(glmprob, na.rm=TRUE),
            sdX0 = sd(glmprob, na.rm=TRUE))
logit_pred_full_tps_agg
logit_pred_full_tps_agg$Variable_List = "Full"

head(logit_pred_grp_tps)
nrow(logit_pred_grp_tps)
logit_pred_grp_tps_agg = logit_pred_grp_tps %>% 
  group_by(synthetic_recnum, target) %>%
  summarize(n = n(),
            mX1 = mean(glmprob, na.rm=TRUE),
            sdX0 = sd(glmprob, na.rm=TRUE))
logit_pred_grp_tps_agg
logit_pred_grp_tps_agg$Variable_List = "Grouped"

head(logit_pred_clin1_tps)
nrow(logit_pred_clin1_tps)
logit_pred_clin1_tps_agg = logit_pred_clin1_tps %>% 
  group_by(synthetic_recnum, target) %>%
  summarize(n = n(),
            mX1 = mean(glmprob, na.rm=TRUE),
            sdX0 = sd(glmprob, na.rm=TRUE))
logit_pred_clin1_tps_agg
logit_pred_clin1_tps_agg$Variable_List = "Clinically Guided"
nrow(logit_pred_clin1_tps_agg)
head(logit_pred_clin1_tps_agg)

######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
#GRADIENT BOOSTING WITH PENALIZATION
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################


##################################################################
##################################################################
##################################################################
#Tune tree-specific parameters ( max_depth, min_child_weight, gamma, subsample, colsample_bytree) 
#    for decided learning rate and number of trees.
##################################################################
##################################################################
##################################################################

tuneparms=expand.grid(eta=c(.1, .01), 
                      nrounds=500, 
                      objective="binary:logistic", 
                      verbose=1, 
                      eval_metric_auc="auc",
                      eval_metric_logloss="logloss",
                      eval_metric_error="error",
                      max_depth=7,
                      min_child_weight=c(.1, .01),
                      alpha=1,
                      subsample=1,
                      colsample_bytree=1,
                      gamma=c(.01, 0)
                      )
tuneparms$tuneparms_rowid = 1:nrow(tuneparms)
#tuneparms$Description = paste0("Eta = ", tuneparms$eta, " Max. Depth = ", tuneparms$max_depth,  " Min. Child Wt. = ", tuneparms$min_child_weight, " Gamma = ", tuneparms$gamma)
tuneparms$Description = paste0("Eta=", tuneparms$eta, " Min. Child Wt.=", tuneparms$min_child_weight, " Gamma=", tuneparms$gamma)
tuneparms
nrow(tuneparms)

#######################################################
#######################################################
#######################################################
#full predictor XGB
#######################################################
#######################################################
#######################################################

# #loop labels
# i = 1 #tuning paramaters
# j = 1 #sample split
# k = 1 #fold number

xgb_undersampling_k_fold_cv = function(FEATURE_SPACE_FORMULA,
                                       OUTCOME_TEXT,
                                       INPUT_DATA,
                                       SAMPLES_IN_INPUT_DATA,
                                       FOLDS_IN_INPUT_DATA){
  trainmod = list()
  xgbmodtree = list()
  trainimpmat = list()
  testpred = list()
  trainmod_folds=list()
  trainimpmat_folds=list()
  testpred_folds = list()
  trainmod_foldtunes=list()
  trainimpmat_foldtunes=list()
  testpred_foldtunes = list()
  for(i in 1:nrow(tuneparms)){
    for(j in 1:SAMPLES_IN_INPUT_DATA){
      for(k in 1:FOLDS_IN_INPUT_DATA){
        #put in to sparse matrices for algorithms
        train_sparse_matrix = sparse.model.matrix(as.formula(paste0(OUTCOME_TEXT, "~ -1 + ", FEATURE_SPACE_FORMULA)), data=INPUT_DATA[[j]]$train[[k]])
        test_sparse_matrix = sparse.model.matrix(as.formula(paste0(OUTCOME_TEXT, "~ -1 + ", FEATURE_SPACE_FORMULA)), data=INPUT_DATA[[j]]$test[[k]])
        trainlabel = INPUT_DATA[[j]]$train[[k]][,grepl(OUTCOME_TEXT, colnames(INPUT_DATA[[j]]$train[[k]]))]
        testlabel = INPUT_DATA[[j]]$test[[k]][,grepl(OUTCOME_TEXT, colnames(INPUT_DATA[[j]]$test[[k]]))]
        trainlist = list(data=train_sparse_matrix, label=trainlabel)
        testlist = list(data=test_sparse_matrix, label=testlabel)
        dtrain = xgb.DMatrix(data = trainlist$data, label = trainlist$label)
        dtest = xgb.DMatrix(data = testlist$data, label = testlist$label)
        #watchlist for early stopping
        error_watch_list = list(train=dtrain, test=dtest)
        trainmod[[k]] = xgb.train(data=dtrain,
                                  eta=tuneparms$eta[i],
                                  nrounds=tuneparms$nrounds[i],
                                  objective=tuneparms$objective[i],
                                  verbose=tuneparms$verbose[i],
                                  eval_metric=tuneparms$eval_metric_auc[i],
                                  eval_metric=tuneparms$eval_metric_error[i],
                                  eval_metric=tuneparms$eval_metric_error[i],
                                  max_depth=tuneparms$max_depth[i],
                                  min_child_weight=tuneparms$min_child_weight[i],
                                  alpha=tuneparms$alpha[i],
                                  subsample=tuneparms$subsample[i],
                                  colsample_bytree=tuneparms$colsample_bytree[i],
                                  gamma=tuneparms$gamma[i])
        #need xgboost vs xgb.train for xgbmodtree object return?
        #    xgbmodtree[[k]] = xgb.model.dt.tree(trainmod[[k]])
        trainimpmat[[k]] = cbind.data.frame(
          data.frame(
            xgb.importance(
              feature_names = dimnames(train_sparse_matrix)[[2]], model=trainmod[[k]])), 
              tune_parm_set_index = tuneparms$tuneparms_rowid[i])
        trainimpmat[[k]]$Sample = j
        trainimpmat[[k]]$Fold = k
        testpred[[k]] = cbind.data.frame(
          data.frame(pred = predict(trainmod[[k]], dtest), 
                     target=testlabel,
                     Sample = j,
                     Fold = k,
                     INPUT_DATA[[j]]$test[[k]])) 
        testpred[[k]]$tune_parm_set_index = tuneparms$tuneparms_rowid[i]
      }
      trainmod_folds[[j]] = list(trainmod)
      trainimpmat_folds[[j]] = do.call("rbind.data.frame", trainimpmat)
      testpred_folds[[j]] = do.call("rbind.data.frame", testpred)
    }
    trainmod_foldtunes[[i]]=list(trainmod_folds)
    trainimpmat_foldtunes[[i]]=do.call("rbind.data.frame", trainimpmat_folds)
    testpred_foldtunes[[i]]=do.call("rbind.data.frame", testpred_folds)
  }
  #don't return training models due to space issues (see commented code if need to retain smaller models)
  #retlist=list(train_models = trainmod_foldtunes, traingain = trainimpmat_foldtunes, testpreds=testpred_foldtunes)
  retlist=list(traingain = trainimpmat_foldtunes, testpreds=testpred_foldtunes)
}

xgb_full_starttime = proc.time()
xgb_full = xgb_undersampling_k_fold_cv(FEATURE_SPACE_FORMULA = ". - synthetic_recnum",
                                      OUTCOME_TEXT = "target",
                                      INPUT_DATA = sims_tps,
                                      SAMPLES_IN_INPUT_DATA=length(sims_tps),
                                      FOLDS_IN_INPUT_DATA=global_k_folds)
xgb_full_endtime = proc.time()
xgb_full_runtime = (xgb_full_endtime-xgb_full_starttime)/3600
xgb_full_runtime

xgb_grp_starttime = proc.time()
xgb_grp = xgb_undersampling_k_fold_cv(FEATURE_SPACE_FORMULA = ". - synthetic_recnum",
                            OUTCOME_TEXT = "target",
                            INPUT_DATA = grp_sims_tps,
                            SAMPLES_IN_INPUT_DATA=length(grp_sims_tps),
                            FOLDS_IN_INPUT_DATA=global_k_folds)
xgb_grp_endtime = proc.time()
xgb_grp_runtime = (xgb_grp_endtime-xgb_grp_starttime)/3600
xgb_grp_runtime


xgb_clin1_starttime = proc.time()
xgb_clin1 = xgb_undersampling_k_fold_cv(FEATURE_SPACE_FORMULA = ". - synthetic_recnum",
                                      OUTCOME_TEXT = "target",
                                      INPUT_DATA = clin1_sims_tps,
                                      SAMPLES_IN_INPUT_DATA=length(clin1_sims_tps),
                                      FOLDS_IN_INPUT_DATA=global_k_folds)
xgb_clin1_endtime = proc.time()
xgb_clin1_runtime = (xgb_clin1_endtime-xgb_clin1_starttime)/3600
xgb_clin1_runtime


##############################################################
##############################################################
##############################################################
#averaging Information Gain esstimates across repeated k-fold cross validation AND tuning sets (for comparison with other methods' variable importance estimates)
##############################################################
##############################################################
##############################################################

var_gb_full_nt = do.call("rbind.data.frame", xgb_full$traingain)
colnames(var_gb_full_nt) = c("Feature", "Gain", "Cover", "Frequency", "tune_parm_set_index", "Sample", "Fold")
var_gb_grp_nt = do.call("rbind.data.frame", xgb_grp$traingain)
colnames(var_gb_grp_nt) = c("Feature", "Gain", "Cover", "Frequency", "tune_parm_set_index", "Sample", "Fold")
var_gb_clin1_nt = do.call("rbind.data.frame", xgb_clin1$traingain)
colnames(var_gb_clin1_nt) = c("Feature", "Gain", "Cover", "Frequency", "tune_parm_set_index", "Sample", "Fold")


var_gb_full = merge(var_gb_full_nt, tuneparms, by.x=c("tune_parm_set_index"), by.y=c("tuneparms_rowid"), all.x=TRUE, all.y=TRUE)
var_gb_grp = merge(var_gb_grp_nt, tuneparms, by.x=c("tune_parm_set_index"), by.y=c("tuneparms_rowid"), all.x=TRUE, all.y=TRUE)
var_gb_clin1 = merge(var_gb_clin1_nt, tuneparms, by.x=c("tune_parm_set_index"), by.y=c("tuneparms_rowid"), all.x=TRUE, all.y=TRUE)


var_gb_full_des = svydesign(id=~Sample + Fold + tune_parm_set_index, weights=~1, data=var_gb_full)
var_gb_grp_des = svydesign(id=~Sample + Fold + tune_parm_set_index, weights=~1, data=var_gb_grp)
var_gb_clin1_des = svydesign(id=~Sample + Fold + tune_parm_set_index, weights=~1, data=var_gb_clin1)


full_var_gb_means = cbind.data.frame(svyby(~Gain, ~Feature + tune_parm_set_index + Description, design=var_gb_full_des, svymean),
                                     confint(svyby(~Gain, ~Feature + tune_parm_set_index + Description, design=var_gb_full_des, svymean)))
full_var_gb_means$Variable_Set = "Full List"

grp_var_gb_means = cbind.data.frame(svyby(~Gain, ~Feature + tune_parm_set_index + Description, design=var_gb_grp_des, svymean),
                                    confint(svyby(~Gain, ~Feature + tune_parm_set_index + Description, design=var_gb_grp_des, svymean)))
grp_var_gb_means$Variable_Set = "Grouped List"

clin1_var_gb_means = cbind.data.frame(svyby(~Gain, ~Feature + tune_parm_set_index + Description, design=var_gb_clin1_des, svymean),
                                    confint(svyby(~Gain, ~Feature + tune_parm_set_index + Description, design=var_gb_clin1_des, svymean)))
clin1_var_gb_means$Variable_Set = "Clinically Informed List"

var_gb_means = rbind.data.frame(full_var_gb_means, grp_var_gb_means, clin1_var_gb_means)
var_gb_means$rowid = 1:nrow(var_gb_means)
colnames(var_gb_means) = c("Feature", "tune_parm_set_index", "Description", "Gain", "se", "LL95", "UL95", "Variable_Set", "rowid")
var_gb_means$new_featurenames = gsub("[...]", " ", sub("X", "", var_gb_means$Feature))
head(var_gb_means)

dev.off()
var_gb_means_plot = ggplot(var_gb_means) +
  geom_hline(yintercept=0, colour="black", size=.3) +
  geom_errorbar(aes(x=reorder(new_featurenames, -Gain), ymin=LL95, ymax=UL95), width=0, size=1) +
  geom_point(data=var_gb_means, aes(x=reorder(new_featurenames, -Gain), y=Gain, colour=Variable_Set), size=1) +
  coord_flip() +
  scale_colour_manual(values = c("firebrick", "navy", "orange"), name="Variable Set") +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Information Gain") +
  facet_wrap(~Description, nrow=2) +
  ylab("Mean Variable Gain") +
  xlab("Variable") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=8, face="bold"),
        strip.text.y = element_text(size=8, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=10),
        axis.text.y=element_text(size=10),
        panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.minor = element_blank(),
        legend.position = "bottom")
var_gb_means_plot
ggsave("/RPROJECTS/Totapally_PICU_KID_EPLOS/variable_gain_from_rep_k_fold_xgb.pdf", var_gb_means_plot, width=11, height=8, scale=1.4)

######################################################
#based on tuning parameter set performance, select set 1 for variable gain estimates
######################################################

selected_var_gb_means = subset(var_gb_means, tune_parm_set_index==1)

selected_var_gb_means_plot = ggplot(selected_var_gb_means) +
  geom_hline(yintercept=0, colour="black", size=.3) +
  geom_errorbar(aes(x=reorder(new_featurenames, -Gain), ymin=LL95, ymax=UL95), width=0, size=1) +
  geom_point(data=selected_var_gb_means, aes(x=reorder(new_featurenames, -Gain), y=Gain, colour=Variable_Set), size=1) +
  coord_flip() +
  scale_colour_manual(values = c("firebrick", "navy", "orange"), name="Variable Set") +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Information Gain") +
  facet_wrap(~Variable_Set, nrow=1) +
  ylab("Mean Variable Gain") +
  xlab("Variable") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=8, face="bold"),
        strip.text.y = element_text(size=8, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=10),
        axis.text.y=element_text(size=10),
        panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.minor = element_blank(),
        legend.position = "bottom")
selected_var_gb_means_plot
ggsave("/RPROJECTS/Totapally_PICU_KID_EPLOS/best_xgb_variable_gain_from_rep_k_fold_xgb.pdf", selected_var_gb_means_plot, width=11, height=8, scale=1.4)


######################################################
######################################################
######################################################
#xgb K-fold-cross-validated under-sampled predictions
######################################################
######################################################
######################################################


xgb_pred_full_tps = subset(do.call("rbind.data.frame", xgb_full$testpreds), select=-c(target.1))
xgb_pred_grp_tps = subset(do.call("rbind.data.frame", xgb_grp$testpreds), select=-c(target.1))
xgb_pred_clin1_tps = subset(do.call("rbind.data.frame", xgb_clin1$testpreds), select=-c(target.1))

summary(xgb_pred_full_tps)
summary(xgb_pred_grp_tps)
summary(xgb_pred_clin1_tps)

head(xgb_pred_full_tps)
nrow(xgb_pred_full_tps)
xgb_pred_full_tps_agg = xgb_pred_full_tps %>% 
  group_by(synthetic_recnum, tune_parm_set_index, target) %>%
  summarize(n = n(),
            mX1 = mean(pred, na.rm=TRUE),
            sdX0 = sd(pred, na.rm=TRUE))
xgb_pred_full_tps_agg
xgb_pred_full_tps_agg$Variable_List = "Full"

head(xgb_pred_grp_tps)
nrow(xgb_pred_grp_tps)
xgb_pred_grp_tps_agg = xgb_pred_grp_tps %>% 
  group_by(synthetic_recnum, tune_parm_set_index, target) %>%
  summarize(n = n(),
            mX1 = mean(pred, na.rm=TRUE),
            sdX0 = sd(pred, na.rm=TRUE))
xgb_pred_grp_tps_agg
xgb_pred_grp_tps_agg$Variable_List = "Grouped"

head(xgb_pred_clin1_tps)
nrow(xgb_pred_clin1_tps)
xgb_pred_clin1_tps_agg = xgb_pred_clin1_tps %>% 
  group_by(synthetic_recnum, tune_parm_set_index, target) %>%
  summarize(n = n(),
            mX1 = mean(pred, na.rm=TRUE),
            sdX0 = sd(pred, na.rm=TRUE))
xgb_pred_clin1_tps_agg
xgb_pred_clin1_tps_agg$Variable_List = "Clinically Informed"


######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
#BINARY CLASSIFIER PREDICTION PERFORMANCE METRICS
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################

binary_classifier_performance = function(BINARY_TARGET, BINARY_PREDICTION, CLASSIFIER_DESCRIPTION_TEXT=NA, CUTOFF_MIN=.01, CUTOFF_MAX=.99, RANGE_CUTOFFS=.01){
  #define cutoff ranges for prediction performance of binary classifier
  cutoffs = seq(CUTOFF_MIN, CUTOFF_MAX, by=RANGE_CUTOFFS)
  newpred_cutoffs = data.frame(matrix(BINARY_PREDICTION, ncol=length(cutoffs), length(BINARY_PREDICTION)))
  for(i in 1:length(cutoffs)){
    newpred_cutoffs[,i] = ifelse(newpred_cutoffs[,i]>=cutoffs[i], 1, 0)
  }
  #full indicator performance
  newpred_full_confmats = list()
  newpred_full_accuracy_dat = data.frame(cutoff=NA, acc_point = NA, acc_lower=NA, acc_upper=NA)
  newpred_full_performance = list()
  for(i in 1:ncol(newpred_cutoffs)){
    newpred_full_confmats[[i]] = confusionMatrix(newpred_cutoffs[,i], BINARY_TARGET, positive="1", mode="everything")
    newpred_full_accuracy_dat[i,1] = cutoffs[i]
    newpred_full_accuracy_dat[i,2] = newpred_full_confmats[[i]]$overall[1]
    newpred_full_accuracy_dat[i,3] = newpred_full_confmats[[i]]$overall[3]
    newpred_full_accuracy_dat[i,4] = newpred_full_confmats[[i]]$overall[4]
    colnames(newpred_full_accuracy_dat) = c("Cutoff", "Accuracy", "LL95", "UL95")
    newpred_full_performance[[i]] = data.frame(cutoff = cutoffs[i], 
                                               TNR=newpred_full_confmats[[i]]$table[1],
                                               TPR=newpred_full_confmats[[i]]$table[4],
                                               FNR=newpred_full_confmats[[i]]$table[3],
                                               FPR=newpred_full_confmats[[i]]$table[2],
                                               matrix(newpred_full_confmats[[i]]$byClass, nrow=1, ncol=11))
    colnames(newpred_full_performance[[i]]) = c("Cutoff", "TNR", "TPR", "FNR", "FPR", 
                                                "Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value", 
                                                "Precision", "Recall", "F1", "Prevalence", "Detection Rate", 
                                                "Detection Prevalence", "Balanced Accuracy")
  }
  predfullperfdat = do.call("rbind.data.frame", newpred_full_performance)
  perfdatall = cbind.data.frame(Description=rep(CLASSIFIER_DESCRIPTION_TEXT, times=nrow(predfullperfdat)), predfullperfdat, newpred_full_accuracy_dat)
  perfdatall_longform = reshape2::melt(perfdatall, id=c("Description", "Cutoff"))
  perfdatret = list(wideperfs = perfdatall, longperfs = perfdatall_longform)
  return(perfdatret)
}

nnpred_full_perf = binary_classifier_performance(BINARY_TARGET=nnpred_full_tps_agg$target , BINARY_PREDICTION=nnpred_full_tps_agg$mX1, CLASSIFIER_DESCRIPTION_TEXT="NN Full Model") 
nnpred_grp_perf = binary_classifier_performance(BINARY_TARGET=nnpred_grp_tps_agg$target , BINARY_PREDICTION=nnpred_grp_tps_agg$mX1, CLASSIFIER_DESCRIPTION_TEXT="NN Grouped Model") 
nnpred_clin1_perf = binary_classifier_performance(BINARY_TARGET=nnpred_clin1_tps_agg$target , BINARY_PREDICTION=nnpred_clin1_tps_agg$mX1, CLASSIFIER_DESCRIPTION_TEXT="NN Clinically Informed Model") 

logit_pred_full_perf = binary_classifier_performance(BINARY_TARGET=logit_pred_full_tps_agg$target , BINARY_PREDICTION=logit_pred_full_tps_agg$mX1, CLASSIFIER_DESCRIPTION_TEXT="Logistic Full Model") 
logit_pred_grp_perf = binary_classifier_performance(BINARY_TARGET=logit_pred_grp_tps_agg$target , BINARY_PREDICTION=logit_pred_grp_tps_agg$mX1, CLASSIFIER_DESCRIPTION_TEXT="Logistic Grouped Model") 
logit_pred_clin1_perf = binary_classifier_performance(BINARY_TARGET=logit_pred_clin1_tps_agg$target , BINARY_PREDICTION=logit_pred_clin1_tps_agg$mX1, CLASSIFIER_DESCRIPTION_TEXT="Logistic Clinically Informed Model") 


xgb_pred_full_perf_wide = list()
xgb_pred_full_perf_long = list()
for(i in 1:nrow(tuneparms)){
  full_temp = xgb_pred_full_tps_agg[xgb_pred_full_tps_agg$tune_parm_set_index==i,]
  xgb_pred_full_perf = binary_classifier_performance(
    BINARY_TARGET=full_temp$target, 
    BINARY_PREDICTION=full_temp$mX1, 
    CLASSIFIER_DESCRIPTION_TEXT="XGB Full Model") 
  xgb_pred_full_perf_wide[[i]] = xgb_pred_full_perf$wideperfs
  xgb_pred_full_perf_wide[[i]]$Tuning_Set = i
  xgb_pred_full_perf_long[[i]] = xgb_pred_full_perf$longperfs
  xgb_pred_full_perf_long[[i]]$Tuning_Set = i
}
xgb_pred_full_perf_long_dat = do.call("rbind.data.frame", xgb_pred_full_perf_long)
summary(xgb_pred_full_perf_long_dat)

xgb_pred_grp_perf_wide = list()
xgb_pred_grp_perf_long = list()
for(i in 1:nrow(tuneparms)){
  grp_temp = xgb_pred_grp_tps_agg[xgb_pred_grp_tps_agg$tune_parm_set_index==i,]
  xgb_pred_grp_perf = binary_classifier_performance(
    BINARY_TARGET=grp_temp$target, 
    BINARY_PREDICTION=grp_temp$mX1, 
    CLASSIFIER_DESCRIPTION_TEXT="XGB Grouped Model") 
  xgb_pred_grp_perf_wide[[i]] = xgb_pred_grp_perf$wideperfs
  xgb_pred_grp_perf_wide[[i]]$Tuning_Set = i
  xgb_pred_grp_perf_long[[i]] = xgb_pred_grp_perf$longperfs
  xgb_pred_grp_perf_long[[i]]$Tuning_Set = i
}
xgb_pred_grp_perf_long_dat = do.call("rbind.data.frame", xgb_pred_grp_perf_long)
summary(xgb_pred_grp_perf_long_dat)


xgb_pred_clin1_perf_wide = list()
xgb_pred_clin1_perf_long = list()
for(i in 1:nrow(tuneparms)){
  clin1_temp = xgb_pred_clin1_tps_agg[xgb_pred_clin1_tps_agg$tune_parm_set_index==i,]
  xgb_pred_clin1_perf = binary_classifier_performance(
    BINARY_TARGET=clin1_temp$target, 
    BINARY_PREDICTION=clin1_temp$mX1, 
    CLASSIFIER_DESCRIPTION_TEXT="XGB Clinically Informed Model") 
  xgb_pred_clin1_perf_wide[[i]] = xgb_pred_clin1_perf$wideperfs
  xgb_pred_clin1_perf_wide[[i]]$Tuning_Set = i
  xgb_pred_clin1_perf_long[[i]] = xgb_pred_clin1_perf$longperfs
  xgb_pred_clin1_perf_long[[i]]$Tuning_Set = i
}
xgb_pred_clin1_perf_long_dat = do.call("rbind.data.frame", xgb_pred_clin1_perf_long)
summary(xgb_pred_clin1_perf_long_dat)



xgballperflong = rbind.data.frame(xgb_pred_full_perf_long_dat, xgb_pred_grp_perf_long_dat, xgb_pred_clin1_perf_long_dat)
xgballperflongmerge = merge(xgballperflong, tuneparms, by.x="Tuning_Set", by.y="tuneparms_rowid", all.x=TRUE) %>% 
  mutate_if(is.character, as.factor)
summary(xgballperflongmerge)
xgballperflongmergesub=subset(xgballperflongmerge, variable=="Sensitivity"|variable=="Specificity"|variable=="Pos Pred Value"|variable=="Neg Pred Value"|variable=="Accuracy")
head(xgballperflongmergesub)

############################################################
#assess xg tuning paramater performance to select best tuning paramater set
############################################################

xgballperflongmergesub %>% 
  filter(Cutoff>.49 & Cutoff<.51) %>%
  group_by(Description.x, Description.y, variable) %>%
  summarize(Cutoffobs = n(),
            mean = mean(value, na.rm=TRUE),
            sd = sd(value, na.rm=TRUE))

xgbperfmetsrangeplot = ggplot(data=xgballperflongmergesub) +
  geom_vline(xintercept=.5, colour="black", size=.3) +
  geom_line(data=xgballperflongmergesub, aes(x=Cutoff, y=value, group=Description.y, colour=Description.y), size=1) +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  #scale_colour_manual(values=c("firebrick", "navy", "pink", "steelblue"), name="Description") +
  facet_grid(Description.x~variable) +
  ylab("Value") +
  xlab("Cutoff") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.minor = element_blank(),
        legend.position = "bottom")
xgbperfmetsrangeplot
ggsave("/RPROJECTS/Totapally_PICU_KID_EPLOS/xgb_performance_across_tuning_parm_sets.pdf", xgbperfmetsrangeplot, width=11, height=8, scale=1.4)


###################################################################################
###################################################################################
###################################################################################
#SELECT BEST XGB model
###################################################################################
###################################################################################
###################################################################################

#select model with tuning parameters of Eta=.1, Min. Child Wt. =.1, and Gamma = .01
#this is tuning set row 1
SELECTED_TUNESETROW = 1
xgbbestfullpreds = subset(xgb_pred_full_perf_long_dat, xgb_pred_full_perf_long_dat$Tuning_Set==SELECTED_TUNESETROW, select=-c(Tuning_Set))
xgbbestgrppreds = subset(xgb_pred_grp_perf_long_dat, xgb_pred_grp_perf_long_dat$Tuning_Set==SELECTED_TUNESETROW, select=-c(Tuning_Set))
xgbbestclin1preds = subset(xgb_pred_clin1_perf_long_dat, xgb_pred_clin1_perf_long_dat$Tuning_Set==SELECTED_TUNESETROW, select=-c(Tuning_Set))


###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
#plot all range of performance metrics
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################

allperfs = rbind.data.frame(xgbbestfullpreds, xgbbestgrppreds, xgbbestclin1preds,
                            nnpred_full_perf$longperfs, nnpred_grp_perf$longperfs, nnpred_clin1_perf$longperfs,
                            logit_pred_full_perf$longperfs, logit_pred_grp_perf$longperfs, logit_pred_clin1_perf$longperfs)


subperfs = subset(allperfs, variable=="Sensitivity"|variable=="Specificity"|variable=="Pos Pred Value"|variable=="Neg Pred Value"|variable=="Accuracy")
summary(subperfs)

perfmetsrangeplot = ggplot(data=subperfs) +
  geom_vline(xintercept=.5, colour="black", size=.3) +
  geom_line(data=subperfs, aes(x=Cutoff, y=value, group=Description, colour=Description), size=1) +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  scale_colour_manual(values=c("firebrick", "navy", "pink", "yellow4", "steelblue", "red1", "lightblue", "yellow", "orange"), name="Description") +
  facet_wrap(~variable, nrow=1, scales="free") +
  ylab("Value") +
  xlab("Cutoff") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.minor = element_blank(),
        legend.position = "bottom")
perfmetsrangeplot
ggsave("/RPROJECTS/Totapally_PICU_KID_EPLOS/performance_range_among_classifiers.pdf", perfmetsrangeplot, width=11, height=8, scale=1.4)


###################################################################
###################################################################
###################################################################
#Discrimination across thresholds of classification for each learner
###################################################################
###################################################################
###################################################################
citation("pROC")

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

rocnnfull = discrimination_auroc_estimation(BINARY_TARGET=nnpred_full_tps_agg$target, MODEL_PREDICTION=nnpred_full_tps_agg$mX1, MODEL_DESCRIPTION_TEXT="NN Full")
rocnngrp = discrimination_auroc_estimation(nnpred_grp_tps_agg$target, nnpred_grp_tps_agg$mX1, MODEL_DESCRIPTION_TEXT="NN Grouped")
rocnnclin1 = discrimination_auroc_estimation(nnpred_clin1_tps_agg$target, nnpred_clin1_tps_agg$mX1, MODEL_DESCRIPTION_TEXT="NN Clinically Informed")
roclgfull = discrimination_auroc_estimation(logit_pred_full_tps_agg$target, logit_pred_full_tps_agg$mX1, MODEL_DESCRIPTION_TEXT="Logistic Full")
roclggrp = discrimination_auroc_estimation(logit_pred_grp_tps_agg$target, logit_pred_grp_tps_agg$mX1, MODEL_DESCRIPTION_TEXT="Logistic Grouped")
roclgclin1 = discrimination_auroc_estimation(logit_pred_clin1_tps_agg$target, logit_pred_clin1_tps_agg$mX1, MODEL_DESCRIPTION_TEXT="Logistic Clinically Informed")

#XGB parameter set tuned selected
selected_xgb_pred_full_tps_agg = subset(xgb_pred_full_tps_agg, tune_parm_set_index==SELECTED_TUNESETROW)
selected_xgb_pred_grp_tps_agg = subset(xgb_pred_grp_tps_agg, tune_parm_set_index==SELECTED_TUNESETROW)
selected_xgb_pred_clin1_tps_agg = subset(xgb_pred_clin1_tps_agg, tune_parm_set_index==SELECTED_TUNESETROW)



rocxgbfull = discrimination_auroc_estimation(selected_xgb_pred_full_tps_agg$target, selected_xgb_pred_full_tps_agg$mX1, MODEL_DESCRIPTION_TEXT="XGB Full")
rocxgbgrp = discrimination_auroc_estimation(selected_xgb_pred_grp_tps_agg$target, selected_xgb_pred_grp_tps_agg$mX1, MODEL_DESCRIPTION_TEXT="XGB Grouped")
rocxgbclin1 = discrimination_auroc_estimation(selected_xgb_pred_clin1_tps_agg$target, selected_xgb_pred_clin1_tps_agg$mX1, MODEL_DESCRIPTION_TEXT="XGB Clinically Informed")


rocdats = rbind.data.frame(rocnnfull$rocdat, rocnngrp$rocdat, rocnnclin1$rocdat, 
                           roclgfull$rocdat, roclggrp$rocdat, roclgclin1$rocdat,
                           rocxgbfull$rocdat, rocxgbgrp$rocdat, rocxgbclin1$rocdat) %>% mutate_if(is.character, as.factor)
head(rocdats)
summary(rocdats)

discrimsplot = ggplot(rocdats) +
  geom_abline(intercept=0, slope=1, size=1) +
  geom_line(data=rocdats, aes(x=100-specificities, y=sensitivities, group=Full_Description, colour=Full_Description, linetype=Full_Description), size=1) +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  scale_linetype_discrete(name="Model") +
  scale_fill_manual(values=c("firebrick", "navy", "pink", "yellow4", "steelblue", "red1", "lightblue", "yellow", "orange"), name="Model") +
  scale_colour_manual(values=c("firebrick", "navy", "pink", "yellow4", "steelblue", "red1", "lightblue", "yellow", "orange"), name="Model") +
  #facet_wrap(~Full_Description, nrow=1) +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,1)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,1)) +
  scale_x_continuous(limits=c(0, 100)) +
  scale_y_continuous(limits=c(0, 100)) +
  ggtitle("Model Discrimination") +
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
ggsave("/RPROJECTS/Totapally_PICU_KID_EPLOS/classifier_discrimination.pdf", discrimsplot, width=11, height=8, scale=1.4)


#Note that sample sizes result in statistically significant differences even with very small effect size differences between classifiers
#This is the Delong Test to compare AUROCs
#test differences in AUCs between different learners
roc.test(rocnnfull$roc_object, roclgfull$roc_object)
roc.test(rocnngrp$roc_object, roclggrp$roc_object)
roc.test(rocnnfull$roc_object, rocxgbfull$roc_object)
roc.test(roclgfull$roc_object, rocxgbfull$roc_object)
roc.test(rocnngrp$roc_object, rocxgbgrp$roc_object)
roc.test(roclggrp$roc_object, rocxgbgrp$roc_object)

#test differences between full and grouped models
roc.test(rocnnfull$roc_object, rocnngrp$roc_object)
roc.test(roclgfull$roc_object, roclggrp$roc_object)
roc.test(rocxgbfull$roc_object, rocxgbgrp$roc_object)


#test grouped and full against clinically informed (reduced)
roc.test(rocnnfull$roc_object, rocnnclin1$roc_object)
roc.test(roclgfull$roc_object, roclgclin1$roc_object)
roc.test(rocxgbfull$roc_object, rocxgbclin1$roc_object)
roc.test(rocnngrp$roc_object, rocnnclin1$roc_object)
roc.test(roclggrp$roc_object, roclgclin1$roc_object)
roc.test(rocxgbgrp$roc_object, rocxgbclin1$roc_object)

# #compare = roc.test(rocplot1, rocplot2)
# #compp = ifelse(round(as.numeric(compare$p.value), digits=3)<.001, .001, round(as.numeric(compare$p.value), digits=3))
# #text(30, 30, labels=paste0("Delong Test = ", round(compare$statistic, digits=2), ", p<", compp), cex=1.2, col="black")


###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
#Calibration through Hosmer-Lemeshow groupings
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
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


#######################################################
#NN full prediction calibrations
#######################################################
nn_full_hosmers = list()
nn_full_hosmer_predictions = list()
for(i in 1:length(nnpredi_tps)){
  #run hosmer and lemeshow groupings
  nn_full_hosmers[[i]] = hosmer_lemeshow_glm_function(TARGET=nnpredi_tps[[i]]$target, PREDICTION=nnpredi_tps[[i]]$X1, GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                                   PREDICTION_MODEL_DESCRIPTION_TEXT = "NN Full Model")
  nn_full_hosmers[[i]]$predictions$sample = i
  #create datasets of predictions
  nn_full_hosmer_predictions[[i]] = nn_full_hosmers[[i]]$predictions
}
nn_full_hosmer_preds_dat = do.call("rbind.data.frame", nn_full_hosmer_predictions)

#######################################################
#NN grouped prediction calibrations
#######################################################
nn_grp_hosmers = list()
nn_grp_hosmer_predictions = list()
for(i in 1:length(nnpredi_grp_tps)){
  nn_grp_hosmers[[i]] = hosmer_lemeshow_glm_function(TARGET=nnpredi_grp_tps[[i]]$target, PREDICTION=nnpredi_grp_tps[[i]]$X1, GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                                PREDICTION_MODEL_DESCRIPTION_TEXT = "NN Grouped Model")
  nn_grp_hosmers[[i]]$predictions$sample = i
  nn_grp_hosmer_predictions[[i]] = nn_grp_hosmers[[i]]$predictions
}
nn_grp_hosmer_preds_dat = do.call("rbind.data.frame", nn_grp_hosmer_predictions)

#######################################################
#NN Clinically Informed clin1 prediction calibrations
#######################################################

nn_clin1_hosmers = list()
nn_clin1_hosmer_predictions = list()
for(i in 1:length(nnpredi_clin1_tps)){
  nn_clin1_hosmers[[i]] = hosmer_lemeshow_glm_function(TARGET=nnpredi_clin1_tps[[i]]$target, PREDICTION=nnpredi_clin1_tps[[i]]$X1, GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                                       PREDICTION_MODEL_DESCRIPTION_TEXT = "NN Clinically Informed Model")
  nn_clin1_hosmers[[i]]$predictions$sample = i
  nn_clin1_hosmer_predictions[[i]] = nn_clin1_hosmers[[i]]$predictions
}
nn_clin1_hosmer_preds_dat = do.call("rbind.data.frame", nn_clin1_hosmer_predictions)


#######################################################
#logistic full prediction calibrations
#######################################################
logit_full_hosmers = list()
logit_full_hosmer_predictions = list()
for(i in 1:length(nnpredi_tps)){
  #run hosmer and lemeshow groupings
  logit_full_hosmers[[i]] = hosmer_lemeshow_glm_function(TARGET=logit_predi_tps[[i]]$target, PREDICTION=logit_predi_tps[[i]][,1], GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                                      PREDICTION_MODEL_DESCRIPTION_TEXT = "Logistic Full Model")
  logit_full_hosmers[[i]]$predictions$sample = i
  #create datasets of predictions
  logit_full_hosmer_predictions[[i]] = logit_full_hosmers[[i]]$predictions
}
logit_full_hosmer_preds_dat = do.call("rbind.data.frame", logit_full_hosmer_predictions)

#######################################################
#logistic grouped prediction calibrations
#######################################################
logit_grp_hosmers = list()
logit_grp_hosmer_predictions = list()
for(i in 1:length(nnpredi_grp_tps)){
  logit_grp_hosmers[[i]] = hosmer_lemeshow_glm_function(TARGET=logit_predi_grp_tps[[i]]$target, PREDICTION=logit_predi_grp_tps[[i]][,1], GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                                     PREDICTION_MODEL_DESCRIPTION_TEXT = "Logistic Grouped Model")
  logit_grp_hosmers[[i]]$predictions$sample = i
  logit_grp_hosmer_predictions[[i]] = logit_grp_hosmers[[i]]$predictions
}
logit_grp_hosmer_preds_dat = do.call("rbind.data.frame", logit_grp_hosmer_predictions)

#######################################################
#logistic Clinically Informed clin1 prediction calibrations
#######################################################
logit_clin1_hosmers = list()
logit_clin1_hosmer_predictions = list()
for(i in 1:length(nnpredi_clin1_tps)){
  logit_clin1_hosmers[[i]] = hosmer_lemeshow_glm_function(TARGET=logit_predi_clin1_tps[[i]]$target, PREDICTION=logit_predi_clin1_tps[[i]][,1], GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                                          PREDICTION_MODEL_DESCRIPTION_TEXT = "Logistic Clinically Informed Model")
  logit_clin1_hosmers[[i]]$predictions$sample = i
  logit_clin1_hosmer_predictions[[i]] = logit_clin1_hosmers[[i]]$predictions
}
logit_clin1_hosmer_preds_dat = do.call("rbind.data.frame", logit_clin1_hosmer_predictions)


#######################################################
#xgb full prediction calibrations
#######################################################

head(xgb_full$testpreds[SELECTED_TUNESETROW][[1]])
selected_xgbpreddat_full = subset(xgb_full$testpreds[SELECTED_TUNESETROW][[1]], tune_parm_set_index==SELECTED_TUNESETROW)

length(unique(selected_xgbpreddat_full$Sample))

xgb_full_hosmers = list()
xgb_full_hosmer_predictions = list()
for(i in 1:length(unique(selected_xgbpreddat_full$Sample))){
  #run hosmer and lemeshow groupings
  xgb_full_hosmers[[i]] = hosmer_lemeshow_glm_function(TARGET=selected_xgbpreddat_full$target[selected_xgbpreddat_full$Sample==i],
                                                       PREDICTION=selected_xgbpreddat_full$pred[selected_xgbpreddat_full$Sample==i], 
                                                       GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                                       PREDICTION_MODEL_DESCRIPTION_TEXT = "Xgb Full Model")
  xgb_full_hosmers[[i]]$predictions$sample = i
  #create datasets of predictions
  xgb_full_hosmer_predictions[[i]] = xgb_full_hosmers[[i]]$predictions
}
xgb_full_hosmer_preds_dat = do.call("rbind.data.frame", xgb_full_hosmer_predictions)

#######################################################
#xgb grouped prediction calibrations
#######################################################

head(xgb_grp$testpreds[SELECTED_TUNESETROW][[1]])
selected_xgbpreddat_grp = subset(xgb_grp$testpreds[SELECTED_TUNESETROW][[1]], tune_parm_set_index==SELECTED_TUNESETROW)

length(unique(selected_xgbpreddat_grp$Sample))

xgb_grp_hosmers = list()
xgb_grp_hosmer_predictions = list()
for(i in 1:length(unique(selected_xgbpreddat_grp$Sample))){
  #run hosmer and lemeshow groupings
  xgb_grp_hosmers[[i]] = hosmer_lemeshow_glm_function(TARGET=selected_xgbpreddat_grp$target[selected_xgbpreddat_grp$Sample==i],
                                                      PREDICTION=selected_xgbpreddat_grp$pred[selected_xgbpreddat_grp$Sample==i], 
                                                      GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                                      PREDICTION_MODEL_DESCRIPTION_TEXT = "Xgb Grouped Model")
  xgb_grp_hosmers[[i]]$predictions$sample = i
  #create datasets of predictions
  xgb_grp_hosmer_predictions[[i]] = xgb_grp_hosmers[[i]]$predictions
}
xgb_grp_hosmer_preds_dat = do.call("rbind.data.frame", xgb_grp_hosmer_predictions)

#######################################################
#xgb Clinically Informed clin1 prediction calibrations
#######################################################

head(xgb_clin1$testpreds[SELECTED_TUNESETROW][[1]])
selected_xgbpreddat_clin1 = subset(xgb_clin1$testpreds[SELECTED_TUNESETROW][[1]], tune_parm_set_index==SELECTED_TUNESETROW)

length(unique(selected_xgbpreddat_clin1$Sample))

xgb_clin1_hosmers = list()
xgb_clin1_hosmer_predictions = list()
for(i in 1:length(unique(selected_xgbpreddat_clin1$Sample))){
  #run hosmer and lemeshow groupings
  xgb_clin1_hosmers[[i]] = hosmer_lemeshow_glm_function(TARGET=selected_xgbpreddat_clin1$target[selected_xgbpreddat_clin1$Sample==i],
                                                        PREDICTION=selected_xgbpreddat_clin1$pred[selected_xgbpreddat_clin1$Sample==i], 
                                                        GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                                        PREDICTION_MODEL_DESCRIPTION_TEXT = "Xgb Clinically Informed Model")
  xgb_clin1_hosmers[[i]]$predictions$sample = i
  #create datasets of predictions
  xgb_clin1_hosmer_predictions[[i]] = xgb_clin1_hosmers[[i]]$predictions
}
xgb_clin1_hosmer_preds_dat = do.call("rbind.data.frame", xgb_clin1_hosmer_predictions)

#######################################################
#plot calibrations
#######################################################

calibs = rbind.data.frame(subset(nn_full_hosmer_preds_dat, select=c(original_pred_group_mean, group_p, group_ll, group_ul, Model, sample)),
                          subset(nn_grp_hosmer_preds_dat, select=c(original_pred_group_mean, group_p, group_ll, group_ul, Model, sample)),
                          subset(nn_clin1_hosmer_preds_dat, select=c(original_pred_group_mean, group_p, group_ll, group_ul, Model, sample)),
                          subset(logit_full_hosmer_preds_dat, select=c(original_pred_group_mean, group_p, group_ll, group_ul, Model, sample)),
                          subset(logit_grp_hosmer_preds_dat, select=c(original_pred_group_mean, group_p, group_ll, group_ul, Model, sample)),
                          subset(logit_clin1_hosmer_preds_dat, select=c(original_pred_group_mean, group_p, group_ll, group_ul, Model, sample)),
                          subset(xgb_full_hosmer_preds_dat, select=c(original_pred_group_mean, group_p, group_ll, group_ul, Model, sample)),
                          subset(xgb_grp_hosmer_preds_dat, select=c(original_pred_group_mean, group_p, group_ll, group_ul, Model, sample)),
                          subset(xgb_clin1_hosmer_preds_dat, select=c(original_pred_group_mean, group_p, group_ll, group_ul, Model, sample)))

levels(factor(calibs$Model))

netcalibsplot = ggplot(calibs) +
  geom_abline(intercept=0, slope=1, size=1) +
  #geom_hline(yintercept=0, colour="black", size=.3) +
  #geom_ribbon(aes(x=original_pred_group_mean, ymin=group_ll, ymax=group_ul, fill=Model), alpha=.25) +
  #geom_errorbar(aes(x=original_pred_group_mean, ymin=group_ll, ymax=group_ul, group=Model), width=0, size=1) +
  geom_line(data=calibs, aes(x=original_pred_group_mean, y=group_p, group=sample, colour=Model), size=.04) +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  scale_fill_manual(values=c("firebrick", "navy", "pink", "yellow4", "steelblue", "red1", "lightblue", "yellow", "orange"), name="Model") +
  scale_colour_manual(values=c("firebrick", "navy", "pink", "yellow4", "steelblue", "red1", "lightblue", "yellow", "orange"), name="Model") +
  facet_wrap(~Model, nrow=3) +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,0)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,0)) +
  scale_x_continuous(limits=c(0, 1)) +
  scale_y_continuous(limits=c(0, 1)) +
  ggtitle("Model Calibration (10 Groupings) Across K-Fold Cross-Validation Under-Sampling Iterations") +
  ylab("Observed Probability") +
  xlab("Prediction") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.minor = element_blank(),
        legend.position = "none")
netcalibsplot
ggsave("/RPROJECTS/Totapally_PICU_KID_EPLOS/calibration_among_classifiers.pdf", netcalibsplot, width=11, height=8, scale=1.4)

######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
#Create a prediction set for export (e.g., CSV) and evaluate its performance against raw predictions from XGBoost
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################

#NOTE! can't expand grid for all values of feature space in 63 predictors (too large)
#experimental 
#Use Framingham scoring method from Sullivan et al. Stat Med. 2004
#first evaluate directionality of predictions for feature

######################################################
######################################################
######################################################
#Full Feature Set Sullivan adaptation scoring
######################################################
######################################################
######################################################

head(selected_xgb_pred_full_tps_agg)
head(full)

bestfullpreds = merge(subset(selected_xgb_pred_full_tps_agg, select=c(synthetic_recnum, target, mX1)), 
                      tunesamp, by=c("synthetic_recnum"))
matrix_bestfullpreds = data.frame(model.matrix(~., data=bestfullpreds))[,-1]
all.equal(nrow(matrix_bestfullpreds), nrow(selected_xgb_pred_full_tps_agg), nrow(tunesamp))
names(matrix_bestfullpreds)
ncol(matrix_bestfullpreds)

#variable directionality on prediction from xgboosted model
fulldirect_dat = data.frame(feature = names(matrix_bestfullpreds[,c(5:70)]), spearman = NA, glm_invlogit_pred = NA, spearman_sign = NA, glm_invlogit_sign = NA)
for(i in 5:70){
  fulldirect_dat[i-4,2] = cor(matrix_bestfullpreds[,3], matrix_bestfullpreds[,i], method="spearman")
  fulldirect_dat[i-4,3] = summary(glm(boot::logit(matrix_bestfullpreds[,3])~matrix_bestfullpreds[,i], data=matrix_bestfullpreds))$coefficients[2,1]
  fulldirect_dat[i-4,4] = sign(fulldirect_dat[i-4,2])
  fulldirect_dat[i-4,5] = sign(fulldirect_dat[i-4,3])
}
fulldirect_dat = fulldirect_dat %>% arrange(feature)
summary(fulldirect_dat)
head(fulldirect_dat)

full_scaling_gain = cbind.data.frame(subset(full_var_gb_means, Variable_Set=="Full List" & tune_parm_set_index==1), fulldirect_dat[,-c(2:3)])
full_scaling_gain$min_abs_val = min(abs(full_scaling_gain$Gain))
full_scaling_gain$raw_sullivan = (full_scaling_gain$Gain/full_scaling_gain$min_abs_val)*full_scaling_gain$spearman_sign
full_scaling_gain$sullivan_points = round(full_scaling_gain$raw_sullivan, digits=0)
full_scaling_gain
write.csv(full_scaling_gain, "full_predictor_set_scale_scoring.csv")


full_selected_features = subset(matrix_bestfullpreds, select=sort(dput(names(subset(matrix_bestfullpreds, select=-c(synthetic_recnum, target, mX1, elos))))))
names(full_selected_features)

ncol(full_selected_features)
nrow(full_scaling_gain)
full_scored = cbind.data.frame(subset(matrix_bestfullpreds, select=c(synthetic_recnum, target, mX1, elos)),
                               full_sullivan_score = as.matrix(full_selected_features) %*% as.vector(full_scaling_gain$sullivan_points),
                               full_selected_features)

head(full_scored)
summary(full_scored)


######################################################
######################################################
######################################################
#Clinically Informed Feature Set
######################################################
######################################################
######################################################

head(selected_xgb_pred_clin1_tps_agg)
head(clin1)

bestclin1preds = merge(subset(selected_xgb_pred_clin1_tps_agg, select=c(synthetic_recnum, target, mX1)), clin1, by=c("synthetic_recnum"))
all.equal(nrow(bestclin1preds), nrow(selected_xgb_pred_clin1_tps_agg), nrow(clin1))
names(bestclin1preds)

#variable directionality on prediction from xgboosted model
direct_dat = data.frame(feature = names(bestclin1preds[,5:20]), spearman = NA, glm_invlogit_pred = NA, spearman_sign = NA, glm_invlogit_sign = NA)
for(i in 5:20){
  direct_dat[i-4,2] = cor(bestclin1preds[,3], bestclin1preds[,i], method="spearman")
  direct_dat[i-4,3] = summary(glm(boot::logit(bestclin1preds[,3])~bestclin1preds[,i], data=bestclin1preds))$coefficients[2,1]
  direct_dat[i-4,4] = sign(direct_dat[i-4,2])
  direct_dat[i-4,5] = sign(direct_dat[i-4,3])
}
direct_dat = direct_dat %>% arrange(feature)


clin1_scaling_gain = cbind.data.frame(subset(selected_var_gb_means, Variable_Set=="Clinically Informed List"), direct_dat[,-c(2:3)])
clin1_scaling_gain$min_abs_val = min(abs(clin1_scaling_gain$Gain))
clin1_scaling_gain$raw_sullivan = (clin1_scaling_gain$Gain/clin1_scaling_gain$min_abs_val)*clin1_scaling_gain$spearman_sign
clin1_scaling_gain$sullivan_points = round(clin1_scaling_gain$raw_sullivan, digits=0)
clin1_scaling_gain
write.csv(clin1_scaling_gain, "clinically_informed_scale_scoring.csv")



clin1_selected_features = subset(bestclin1preds, select=sort(dput(names(subset(bestclin1preds, select=-c(synthetic_recnum, target, mX1, elos))))))
names(clin1_selected_features)

ncol(clin1_selected_features)
nrow(clin1_scaling_gain)
clin1_scored = cbind.data.frame(subset(bestclin1preds, select=c(synthetic_recnum, target, mX1, elos)),
                        clin1_sullivan_score = as.matrix(clin1_selected_features) %*% as.vector(clin1_scaling_gain$sullivan_points),
                        clin1_selected_features)

head(clin1_scored)
summary(clin1_scored)

######################################################
######################################################
######################################################
#Sullivan Scored Discrimination 
######################################################
######################################################
######################################################

######################################################
#xgb Full feature prediction discriminations
######################################################

#Sullivan vs raw discrimination
rocfull_scored_raw_pred = discrimination_auroc_estimation(full_scored$target, full_scored$mX1, MODEL_DESCRIPTION_TEXT="XGB Full Predictor Raw Prediction")
rocfull_scored_sullivan = discrimination_auroc_estimation(full_scored$target, full_scored$full_sullivan_score, MODEL_DESCRIPTION_TEXT="XGB Full Predictor Sullivan Score")

roc_full_scoring = rbind.data.frame(rocfull_scored_sullivan$rocdat, rocfull_scored_raw_pred$rocdat) %>% mutate_if(is.character, as.factor)
head(roc_full_scoring)
summary(roc_full_scoring)


#This is the Delong Test to compare AUROCs
#test differences in AUCs between different learners
roc.test(rocfull_scored_sullivan$roc_object, rocfull_scored_raw_pred$roc_object)

######################################################
#xgb Clinically Informed clin1 prediction discriminations
######################################################

#Sullivan vs raw discrimination
rocclin1_scored_raw_pred = discrimination_auroc_estimation(clin1_scored$target, clin1_scored$mX1, MODEL_DESCRIPTION_TEXT="XGB Clinically Informed Raw Prediction")
rocclin1_scored_sullivan = discrimination_auroc_estimation(clin1_scored$target, clin1_scored$clin1_sullivan_score, MODEL_DESCRIPTION_TEXT="XGB Clinically Informed Sullivan Score")

roc_clin1_scoring = rbind.data.frame(rocclin1_scored_sullivan$rocdat, rocclin1_scored_raw_pred$rocdat) %>% mutate_if(is.character, as.factor)
head(roc_clin1_scoring)
summary(roc_clin1_scoring)

roc_scoring_all = rbind.data.frame(rocclin1_scored_sullivan$rocdat, rocfull_scored_sullivan$rocdat)

scored_discrimsplot = ggplot(roc_scoring_all) +
  geom_abline(intercept=0, slope=1, size=1) +
  geom_line(aes(x=100-specificities, y=sensitivities, group=Full_Description, colour=Full_Description, linetype=Full_Description), size=1) +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  scale_linetype_manual(values=c("dotdash", "solid"), name="Scoring") +
  scale_colour_manual(values=c("firebrick", "navy"), name="Scoring") +
  #facet_wrap(~Full_Description, nrow=1) +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,1)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,1)) +
  scale_x_continuous(limits=c(0, 100)) +
  scale_y_continuous(limits=c(0, 100)) +
  ggtitle("Full vs. Clinically Informed Reduced Predictor Sullivan Scoring Discrimination") +
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
        legend.position = "bottom")
scored_discrimsplot
ggsave("/RPROJECTS/Totapally_PICU_KID_EPLOS/sullivan_scored_classifier_discrimination.pdf", scored_discrimsplot, width=11, height=8, scale=1.4)

#This is the Delong Test to compare AUROCs
#test differences in AUCs between different learners
roc.test(rocclin1_scored_sullivan$roc_object, rocclin1_scored_raw_pred$roc_object)


######################################################
#test differences in AUCs between full and clinically informed sullivan scoring
######################################################
roc.test(rocclin1_scored_sullivan$roc_object, rocfull_scored_sullivan$roc_object)

######################################################
######################################################
######################################################
#XGB Sullivan Scored Calibration 
######################################################
######################################################
######################################################

#######################################################
#xgb Full feature prediction calibrations
#######################################################

full_raw_calib = hosmer_lemeshow_glm_function(TARGET=full_scored$target,
                                              PREDICTION=full_scored$mX1, 
                                              GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                              PREDICTION_MODEL_DESCRIPTION_TEXT = "XGB Full Predictors Raw Prediction")

#Note: groups to cut by creates the equal number of 10 categories (Deciles) to parallel the raw calibration
full_sullivan_calib = hosmer_lemeshow_glm_function(TARGET=full_scored$target,
                                                   PREDICTION=full_scored$full_sullivan_score, 
                                                   GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                                   PREDICTION_MODEL_DESCRIPTION_TEXT = "XGB Full Predictors Sullivan Score")
full_sullivan_calib$predictions
full_sullivan_calib

fullscore_calibs = rbind.data.frame(full_raw_calib$predictions, full_sullivan_calib$predictions)
fullscore_calibs

levels(factor(fullscore_calibs$Model))

#######################################################
#xgb Clinically Informed clin1 prediction calibrations
#######################################################

clin1_raw_calib = hosmer_lemeshow_glm_function(TARGET=clin1_scored$target,
                             PREDICTION=clin1_scored$mX1, 
                             GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                             PREDICTION_MODEL_DESCRIPTION_TEXT = "XGB Clinically Informed Raw Prediction")

#Note: groups to cut by creates the equal number of 10 categories (Deciles) to parallel the raw calibration
clin1_sullivan_calib = hosmer_lemeshow_glm_function(TARGET=clin1_scored$target,
                                               PREDICTION=clin1_scored$clin1_sullivan_score, 
                                               GROUPS_TO_CUT_BY=37, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05,
                                               PREDICTION_MODEL_DESCRIPTION_TEXT = "XGB Clinically Informed Sullivan Score")
clin1_sullivan_calib$predictions
clin1_sullivan_calib

clin1score_calibs = rbind.data.frame(clin1_raw_calib$predictions, clin1_sullivan_calib$predictions)
clin1score_calibs

levels(factor(clin1score_calibs$Model))

#############################################
#full Sullivan scoring calibration
#############################################

score_full_calib_plot = ggplot(full_sullivan_calib$predictions) +
  geom_abline(intercept=0, slope=.0225/400, size=1) +
  #geom_hline(yintercept=0, colour="black", size=.3) +
#  geom_ribbon(aes(x=original_pred_group_mean, ymin=group_ll, ymax=group_ul, fill=Model), alpha=.25) +
  #geom_errorbar(aes(x=original_pred_group_mean, ymin=group_ll, ymax=group_ul, group=Model), width=0, size=1) +
  geom_line(aes(x=original_pred_group_mean, y=group_p, colour=Model, linetype=Model), size=1) +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  scale_fill_manual(values=c("firebrick"), name="Scoring") +
  scale_colour_manual(values=c("firebrick"), name="Scoring") +
  #facet_wrap(~Model, nrow=2, scales="free") +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,0)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,0)) +
  #scale_x_continuous(limits=c(-3, 67)) +
  #scale_y_continuous(limits=c(0, 1)) +
  ggtitle("Full Predictor Set\nSullivan Scoring Prediction Calibration") +
  ylab("Observed Probability") +
  xlab("Prediction (10 Bins)") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.minor = element_blank(),
        legend.position = "none")
score_full_calib_plot
ggsave("/RPROJECTS/Totapally_PICU_KID_EPLOS/calibration_full_predictor_sullivan_scoring.pdf", score_full_calib_plot, width=11, height=8, scale=1.4)


#############################################
#clinically informed Sullivan scoring calibration
#############################################
score_clin1_calib_plot = ggplot(clin1_sullivan_calib$predictions) +
  geom_abline(intercept=0, slope=.0525/24.5, size=1) +
  #geom_hline(yintercept=0, colour="black", size=.3) +
#  geom_ribbon(aes(x=original_pred_group_mean, ymin=group_ll, ymax=group_ul, fill=Model), alpha=.25) +
  #geom_errorbar(aes(x=original_pred_group_mean, ymin=group_ll, ymax=group_ul, group=Model), width=0, size=1) +
  geom_line(aes(x=original_pred_group_mean, y=group_p, colour=Model, linetype=Model), size=1) +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  scale_fill_manual(values=c("navy", "firebrick"), name="Scoring") +
  scale_colour_manual(values=c("navy", "firebrick"), name="Scoring") +
  #facet_wrap(~Model, nrow=2, scales="free") +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,0)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,0)) +
  #scale_x_continuous(limits=c(-3, 67)) +
  #scale_y_continuous(limits=c(0, 1)) +
  ggtitle("Clinically Informed\nSullivan Scoring Prediction Calibration") +
  ylab("Observed Probability") +
  xlab("Prediction (10 Bins)") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.minor = element_blank(),
        legend.position = "none")
score_clin1_calib_plot
ggsave("/RPROJECTS/Totapally_PICU_KID_EPLOS/calibration_clinically_informed_sullivan_scoring.pdf", score_clin1_calib_plot, width=11, height=8, scale=1.4)


arrscoreplots = gridExtra::grid.arrange(score_full_calib_plot, score_clin1_calib_plot, ncol=2)
ggsave("/RPROJECTS/Totapally_PICU_KID_EPLOS/arranged_calibration_sullivan_scoring.pdf", arrscoreplots, width=11, height=8, scale=1.4)

# ##FULL PREDICTIONS
# #do full outer join of predictions and observed cases for later averaging
# str(tunesamp)
# head(nnpred_full_tps)
# export_full_model_preds_all = merge(subset(nnpred_full_tps, select=c(X1, synthetic_recnum)), subset(tunesamp, select=c(-elos)), by=c("synthetic_recnum"), all.x=TRUE, all.y=TRUE)
# nrow(export_full_model_preds_all)
# str(export_full_model_preds_all)
# 
# export_full_model_preds = data.frame(export_full_model_preds_all %>% 
#   group_by_at(names(export_full_model_preds_all)[-grep("X1|synthetic_recnum", names(export_full_model_preds_all))]) %>%
#   summarise(Pprior=round(mean(X1, na.rm=TRUE), digits=3), 
#             Psd = sd(X1, na.rm=TRUE),
#             n = n()) %>%
#   arrange(-Pprior))
# names(export_full_model_preds) = gsub("\\.", " ", names(export_full_model_preds))
# nrow(export_full_model_preds)
# head(export_full_model_preds)
# #create posterior that integrates prior knowledge about baseline probability (very low for EPLOS)
# prop.table(table(full$elos))[[2]]
# Full_Posterior_Probability = export_full_model_preds$Pprior*.0005
# summary(export_full_model_preds)
# 
# 
# 
# str(tunesamp)
# head(nnpred_full_tps)
# export_full_model_preds_all = merge(subset(nnpred_full_tps, select=c(X1, synthetic_recnum)), subset(tunesamp, select=c(-elos)), by=c("synthetic_recnum"), all.x=TRUE, all.y=TRUE)
# nrow(export_full_model_preds_all)
# str(export_full_model_preds_all)
# 
# export_full_model_preds = data.frame(export_full_model_preds_all %>% 
#                                        group_by_at(names(export_full_model_preds_all)[-grep("X1|synthetic_recnum", names(export_full_model_preds_all))]) %>%
#                                        summarise(Pprior=round(mean(X1, na.rm=TRUE), digits=3), 
#                                                  Psd = sd(X1, na.rm=TRUE),
#                                                  n = n()) %>%
#                                        arrange(-Pprior))
# names(export_full_model_preds) = gsub("\\.", " ", names(export_full_model_preds))
# nrow(export_full_model_preds)
# head(export_full_model_preds)
# 
# 
# #GROUPED PROCEDURE MODEL
# str(grp_tunesamp)
# head(nnpred_grp_tps)

###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
#assess most informative predictors across classifiers (for Clinically Informed reduced variable set for Dr. Totapally to review; PRE clin1 construction)
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################

# quantile_cut_groups = .33
# 
# #feed forward neural networks variable importance quantiles
# grouped_nn_var_import_means = subset(var_import_means, Variable_Set=="Full List"|Variable_Set=="Grouped List")
# names(grouped_nn_var_import_means)
# grouped_nn_var_import_means$quantile = cut(abs(grouped_nn_var_import_means$importance), breaks=c(quantile(abs(grouped_nn_var_import_means$importance), probs = seq(0, 1, by = quantile_cut_groups), na.rm=TRUE)), include.lowest=TRUE)
# grouped_nn_var_import_means$quantile_num = as.numeric(grouped_nn_var_import_means$quantile)
# table(grouped_nn_var_import_means$quantile, grouped_nn_var_import_means$quantile_num)
# nn_var_quants = subset(grouped_nn_var_import_means, quantile_num==max(grouped_nn_var_import_means$quantile_num, na.rm=TRUE), select=c(Variable_Set, new_coefnames, quantile))
# nn_var_quants$model = "nn"
# nn_var_quants
# 
# #logistic regression coefficients
# grouped_var_coefs_means = subset(var_coefs_means, Variable_Set=="Full List"|Variable_Set=="Grouped List")
# names(grouped_var_coefs_means)
# grouped_var_coefs_means$quantile = cut(abs(grouped_var_coefs_means$beta), breaks=c(quantile(abs(grouped_var_coefs_means$beta), probs = seq(0, 1, by = quantile_cut_groups), na.rm=TRUE)), include.lowest=TRUE)
# grouped_var_coefs_means$quantile_num = as.numeric(grouped_var_coefs_means$quantile)
# table(grouped_var_coefs_means$quantile, grouped_var_coefs_means$quantile_num)
# logit_var_quants = subset(grouped_var_coefs_means, quantile_num==max(grouped_var_coefs_means$quantile_num, na.rm=TRUE), select=c(Variable_Set, new_coefnames, quantile))
# logit_var_quants$model = "logit"
# logit_var_quants
# 
# #xgb gain estimates
# names(var_gb_means)
# selvarmeans = subset(var_gb_means, tune_parm_set_index==1 & (Variable_Set=="Full List"|Variable_Set=="Grouped List"))
# selvarmeans$quantile = cut(abs(selvarmeans$Gain), breaks=c(quantile(abs(selvarmeans$Gain), probs = seq(0, 1, by = quantile_cut_groups), na.rm=TRUE)), include.lowest=TRUE)
# selvarmeans$quantile_num = as.numeric(selvarmeans$quantile)
# table(selvarmeans$quantile, selvarmeans$quantile_num)
# xgb_var_quants = subset(selvarmeans, quantile_num==max(selvarmeans$quantile_num, na.rm=TRUE), select=c(Variable_Set, new_featurenames, quantile))
# colnames(xgb_var_quants) = c("Variable_Set", "new_coefnames", "quantile")
# xgb_var_quants$model = "xgb"
# xgb_var_quants
# 
# varestquantcomp = rbind.data.frame(nn_var_quants, logit_var_quants, xgb_var_quants)
# 
# effect_tertiles = data.frame(table(varestquantcomp$new_coefnames)) %>% arrange(-Freq)
# effect_tertiles
# write.csv(effect_tertiles, "/RPROJECTS/Totapally_PICU_KID_EPLOS/output/effect_tertiles_for_clinical_informed_measure_creation.csv")
# 

#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#Examine L1 penalization approach
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################

# tunesamplass = model.matrix( ~ . -recnum -1 , subset(tunesamp, select=-c(elos)))
# elostunesampfactorvector = factor(tunesamp$elos)
#
# #cvfit_full = cv.glmnet(fullmatlass, elosfullfactorvector, family = "binomial", type.measure = "class", alpha=1, nfolds=5)
# cvfit_tunesamp = cv.glmnet(tunesamplass, elostunesampfactorvector, family = "binomial", type.measure = "class", alpha=1, nfolds=5)
#
# # plot(cvfit_full)
# # cvfit_full$lambda.min
# # cvfit_full$lambda.1se
#
# plot(cvfit_tunesamp)
# cvfit_tunesamp$lambda
# cvfit_tunesamp$lambda.min
# cvfit_tunesamp$lambda.1se
#
# #coef(cvfit_full, s="lambda.min")
# coef(cvfit_tunesamp, s="lambda.min")

######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
#SAVE IMAGE
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################

save.image("/DATA/Totapally_PICU_KID_EPLOS/EPLOS_nnet_cv_V1.RData")

######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
#SCRATCH
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################


# #Note: function below depends on function above (hosmer_lemeshow_glm_function)
# undersampled_kfold_cv_calibration_hosmer_lemeshow=function(UNDERSAMPLED_CV_LIST, 
#                                                            UNDERSAMPLED_CV_TARGET, 
#                                                            UNDERSAMPLED_CV_PREDICTION, 
#                                                            UNDERSAMPLED_CV_GROUPS, 
#                                                            UNDERSAMPLED_CV_P_VALUE=.05, 
#                                                            UNDERSAMPLED_CV_DESCRIPTION=NA){
#   hosmers = list()
#   hosmer_predictions = list()
#   for(i in 1:length(UNDERSAMPLED_CV_LIST)){
#     #run hosmer and lemeshow groupings
#     hosmers[[i]] = hosmer_lemeshow_glm_function(TARGET=UNDERSAMPLED_CV_TARGET, 
#                                                 PREDICTION=UNDERSAMPLED_CV_PREDICTION, 
#                                                 GROUPS_TO_CUT_BY=UNDERSAMPLED_CV_GROUPS, 
#                                                 P_VALUE_FOR_CONFIDENCE_INTERVAL=UNDERSAMPLED_CV_P_VALUE, 
#                                                 PREDICTION_MODEL_DESCRIPTION_TEXT = UNDERSAMPLED_CV_DESCRIPTION)
#     hosmers[[i]]$predictions$sample = i
#     #create datasets of predictions
#     hosmer_predictions[[i]] = hosmers[[i]]$predictions
#   }
#   hosmer_preds_dat = do.call("rbind.data.frame", hosmer_predictions)
#   return(hosmer_preds_dat)
# }
# 
# nn_full_calibs = undersampled_kfold_cv_calibration_hosmer_lemeshow(nnpredi_tps, nnpredi_tps[[i]]$target, nnpredi_tps[[i]]$X1, 10, .05, "Full Model")
# nn_grp_calibs = undersampled_kfold_cv_calibration_hosmer_lemeshow(nnpredi_grp_tps, nnpredi_grp_tps[[i]]$target, nnpredi_grp_tps[[i]]$X1, 10, .05, "Full Model")


#############################################
#calibration plots from GBM with smoothing ##
#############################################

# dev.off()
# par(mfrow=c(1,1))
# gbm::calibrate.plot(nnpred_full_tps$target, nnpred_full_tps$X1, df=9, shade.col = "firebrick1")
# gbm::calibrate.plot(nnpred_grp_tps$target, nnpred_grp_tps$X1, df=9, replace=FALSE, shade.col = "lightblue")

# modified_gbm_calibration = function(y, p, CALIBRATION_MODEL_DESCRIPTION_TEXT = "", df=9, distribution="bernoulli", replace=TRUE, knots=NULL, RANGE_OF_PREDICTIONS_LENGTH=100){
#   data <- data.frame(y = y, p = p)
#   if (is.null(knots) && is.null(df)) 
#     stop("Either knots or df must be specified")
#   if ((df != round(df)) || (df < 1)) 
#     stop("df must be a positive integer")
#   if (distribution == "bernoulli") {
#     family1 = binomial
#   }
#   else if (distribution == "poisson") {
#     family1 = poisson
#   }
#   else {
#     family1 = gaussian
#   }
#   gam1 <- glm(y ~ splines::ns(p, df = df, knots = knots), data = data, 
#               family = family1)
#   x <- seq(min(p), max(p), length = RANGE_OF_PREDICTIONS_LENGTH)
#   yy <- predict(gam1, newdata = data.frame(p = x), se.fit = TRUE, 
#                 type = "response")
#   x <- x[!is.na(yy$fit)]
#   yy$se.fit <- yy$se.fit[!is.na(yy$fit)]
#   yy$fit <- yy$fit[!is.na(yy$fit)]
#   se.lower <- yy$fit - 1.96 * yy$se.fit
#   se.upper <- yy$fit + 1.96 * yy$se.fit
#   if (distribution == "bernoulli") {
#     se.lower[se.lower < 0] <- 0
#     se.upper[se.upper > 1] <- 1
#   }
#   if (distribution == "poisson") {
#     se.lower[se.lower < 0] <- 0
#   }
#   calibration_dat=data.frame(P=x, pred=yy$fit, se=yy$se.fit, ll=se.lower, ul=se.upper)
#   calibration_dat$Description = CALIBRATION_MODEL_DESCRIPTION_TEXT
#   return(calibration_dat)
# }
# 
# fullcalib = modified_gbm_calibration(nnpred_full_tps$target, nnpred_full_tps$X1, CALIBRATION_MODEL_DESCRIPTION_TEXT="Full Model", df=4, RANGE_OF_PREDICTIONS_LENGTH=1000)
# grpcalib = modified_gbm_calibration(nnpred_grp_tps$target, nnpred_grp_tps$X1, CALIBRATION_MODEL_DESCRIPTION_TEXT="Grouped Indicator Model", df=4, RANGE_OF_PREDICTIONS_LENGTH=10000)
