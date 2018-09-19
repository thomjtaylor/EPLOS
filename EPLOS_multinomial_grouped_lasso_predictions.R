

set.seed(57006)

rm(list=ls())

library(caret)
library(plyr)
library(dplyr)
library(lubridate)
library(pROC)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(scales)
library(splines)
library(bindata)

setwd("C:/Users/Owner/Documents/MCHS/ELOS/PHIS/")

#load("~/MCHS/ELOS/PHIS/EPLOS_multinomial_grouped_lasso_predictions.RData")

###################################################
#load data
###################################################

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
summary(xmatall)

ymatall = as.matrix(ohed[, which(names(ohed) %in% grep("LOSmonths", names(ohed), value=TRUE))])
nrow(ymatall); ncol(ymatall)
head(ymatall)
summary(ymatall)

#free up space on local machine, but keep testdat for later since need to prepare test x and y matrices for prediction
#rm(temp, testdat, traindat, ohed)
rm(temp, traindat, ohed)


##########################################################
#one-hot encode test set data frame
##########################################################

testdummyset = dummyVars("~ . ", data=testdat)
testohed = na.omit(data.frame(predict(testdummyset, newdata=testdat)))
nrow(testohed); nrow(testdat)
ncol(testohed)

test_xmatall = as.matrix(testohed[, -which(names(testohed) %in% grep("LOSmonths", names(testohed), value=TRUE))])
nrow(test_xmatall); ncol(test_xmatall)
head(test_xmatall[,1:30])
summary(test_xmatall)

test_ymatall = as.matrix(testohed[, which(names(testohed) %in% grep("LOSmonths", names(testohed), value=TRUE))])
nrow(test_ymatall); ncol(test_ymatall)
head(test_ymatall)
summary(test_ymatall)

##############################################################
##############################################################
##############################################################
#explore glm with L1 penalization, for comparison
##############################################################
##############################################################
##############################################################

yout = ymatall %*% c(0:6)

nrow(weightests$wt)
sum(weightests$wt)
glmnetwt = as.numeric(as.matrix(weightests$wt))

#don't use parallel option when loaded for keras, starts to rerun keras
cvl1mod=glmnet::cv.glmnet(xmatall, yout, family="multinomial", weights=glmnetwt, alpha=1, type.multinomial="grouped", parallel = TRUE)
plot(cvl1mod)
cvl1mod$lambda.min
cvl1mod$lambda.1se
coef(cvl1mod, s="lambda.min")
coef(cvl1mod, s=.04)

str(coef(cvl1mod, s="lambda.min"))
str(coef(cvl1mod, s=.04))

penalization_lambda = .04

#l1mod=glmnet::glmnet(xmatall, ymatall, family="multinomial", weights=glmnetwt, alpha=1, type.multinomial = "grouped")
#l1mod
#plot(l1mod)

#############################################################
#############################################################
#############################################################
#best internval validation model coefficients from each Multinomial LASSO model
#############################################################
#############################################################
#############################################################

#"lambda.1se"
#penalization_lambda

bestlassmod_coefs = do.call("cbind.data.frame", 
                            lapply(coef(cvl1mod, s="lambda.1se"), 
                                   function(x) as.matrix(x)
                            )
)

bestlassdat = data.frame(feature = dimnames(bestlassmod_coefs)[[1]], bestlassmod_coefs)
colnames(bestlassdat)=c("feature", "a. < 1 Month", "b. 1 Month", "c. 2 Months", "d. 3 Months", "e. 4 Months", "f. 5 Months", "g. 6 Months or more")
bestlasslong = reshape2::melt(bestlassdat, id=c("feature")) %>%
  filter(value!=0)

psych::describeBy(bestlasslong$value, bestlasslong$variable, mat=TRUE, digits=2)

#########################################################
#write feature coefficients
#########################################################

#bestlass_coefs_export = subset(bestlasslong, feature!="(Intercept)")
bestlass_coefs_export = bestlasslong
bestlass_intercepts = subset(bestlasslong, feature=="(Intercept)")

bestlasswide = reshape(bestlass_coefs_export, timevar="variable", idvar="feature", direction="wide")
bestlasswide$feature = gsub("_", ".", bestlasswide$feature)
head(bestlasswide)

write.csv(bestlasswide, "eTable4_best_LASSO_multinomial_model_coefficients_for_SQL_pseudocode.csv")

#############################################################
#############################################################
#############################################################
#feasible scoring paramaters (<20) coefficient paramater estimates from multinomial LASSO model
#############################################################
#############################################################
#############################################################

feaslassmod_coefs = do.call("cbind.data.frame", 
                            lapply(coef(cvl1mod, s=penalization_lambda), 
                                   function(x) as.matrix(x)
                            )
)
feaslassdat = data.frame(feature = dimnames(feaslassmod_coefs)[[1]], feaslassmod_coefs)
colnames(feaslassdat)=c("feature", "a. < 1 Month", "b. 1 Month", "c. 2 Months", "d. 3 Months", "e. 4 Months", "f. 5 Months", "g. 6 Months or more")
feaslasslong = reshape2::melt(feaslassdat, id=c("feature")) %>%
  filter(value!=0)

psych::describeBy(feaslasslong$value, feaslasslong$variable, mat=TRUE, digits=2)

#############################################################
#############################################################
#############################################################
#sullivan scoring model with top XX predictors of risk for each Length of Stay Category
#############################################################
#############################################################
#############################################################

SULLIVAN_SCORING_NUMBER_OF_COEFFICEINTS_TO_RETAIN=30

sullfrombest = bestlasslong %>%
  filter(feature!="(Intercept)") %>%
  mutate(absval = abs(value)) %>%
  group_by(variable) %>%
  mutate(order = row_number(-absval)) %>% #includes both negative and positive coefficients of risk
  filter(order<=SULLIVAN_SCORING_NUMBER_OF_COEFFICEINTS_TO_RETAIN) %>%
  arrange(variable, order) %>%
  group_by(variable) %>%
  mutate(mincoef = min(absval, na.rm=TRUE)) %>%
  mutate(raw_sullivan_score = value/mincoef) %>%
  mutate(sullivan_points = round(raw_sullivan_score, digits=0))

sullbestnames = merge(merge(sullfrombest, dxnames, by.x=c("feature"), by.y=c("icdcode"), all.x=TRUE), pxnames, by.x=c("feature"), by.y=c("pxicdcode"), all.x = TRUE)

nrow(sullfrombest); nrow(sullbestnames)
sullbestnames$Description = coalesce(as.character(sullbestnames$icd), as.character(sullbestnames$pxicd))

subsull = subset(sullbestnames, select=c(variable, feature, Description, order, value, raw_sullivan_score, sullivan_points)) %>%
  arrange(-as.numeric(variable), order)

psych::describeBy(subsull$value, group=subsull$variable, mat=TRUE, digits=3)
psych::describeBy(subsull$sullivan_points, group=subsull$variable, mat=TRUE, digits=3)

write.csv(subsull, "eTable5_best_LASSO_sullivan_scoring_by_outcome_and_predictive_feature.csv")


################################################
#point scoring magnitude of risk points visualization
################################################
subsull$neg_sullivan_points = ifelse(subsull$sullivan_points<0, subsull$sullivan_points, NA)
subsull$pos_sullivan_points = ifelse(subsull$sullivan_points>=0, subsull$sullivan_points, NA)

magplot = ggplot(data=subsull, group=1) +
  geom_point(aes(x=reorder(Description, sullivan_points), y=sullivan_points, colour=sullivan_points), size=1) +
  geom_errorbar(aes(x=reorder(Description, sullivan_points), ymin=0, ymax=pos_sullivan_points, colour=sullivan_points), size=2, width=0) +
  geom_errorbar(aes(x=reorder(Description, sullivan_points), ymin=neg_sullivan_points, ymax=0, colour=sullivan_points), size=2, width=0) +
  geom_label(aes(x=reorder(Description, sullivan_points), y=16.5, label=sullivan_points), size=5) +
  #scale_colour_gradient2(low="navy", mid="grey90", high="firebrick", name="Sullivan Based Points Assigned") +
  scale_colour_gradient(low="navy", high="firebrick", name="Sullivan Based Points Assigned") +
  #scale_colour_manual(values=c("gold2", "orange", "orangered", "red", "firebrick1", "firebrick", "firebrick4"), name="Length of Stay") +
  scale_y_continuous(limits=c(-10, 20), breaks=seq(-10, 15, by=5)) +
  facet_wrap(~variable, nrow=1) +
  coord_flip() +
  ggtitle("") +
  ylab("Points Assigned") +
  xlab("Feature") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=14),
        panel.grid.major=element_line(colour="grey80", size=.2),
        #panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "bottom")
#missing bars are NAs created purposely in code above
magplot
ggsave("eFigure1_Sullivan_Points_Scoring_by_Feature_visualization.pdf", magplot, width=11, height=8, scale=2.1)


#############################################################
#prepare scoring on all 2016-2017 CHA PHIS cohort data for estimates of percentile rank of resulting scores
#############################################################

dxnames = read.csv("DX_features_with_at_least_20_neonates_with_EPLOS.csv", header=TRUE)[,-1]
pxnames = read.csv("PX_features_with_at_least_20_neonates_with_EPLOS.csv", header=TRUE)[,-1]
head(dxnames)
head(pxnames)

head(sullfrombest)

ss0 = subset(sullfrombest, variable=="a. < 1 Month", select=c(feature, sullivan_points))
ss1 = subset(sullfrombest, variable=="b. 1 Month", select=c(feature, sullivan_points))
ss2 = subset(sullfrombest, variable=="c. 2 Months", select=c(feature, sullivan_points))
ss3 = subset(sullfrombest, variable=="d. 3 Months", select=c(feature, sullivan_points))
ss4 = subset(sullfrombest, variable=="e. 4 Months", select=c(feature, sullivan_points))
ss5 = subset(sullfrombest, variable=="f. 5 Months", select=c(feature, sullivan_points))
ss6 = subset(sullfrombest, variable=="g. 6 Months or more", select=c(feature, sullivan_points))

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

###########################################################################
###########################################################################
###########################################################################
#Sullivan point scores calibrations
###########################################################################
###########################################################################
###########################################################################

#LINPRED_CUT_THRESHOLDS = 10

sull_score_logit_models = list()
sull_score_logit_predictions = list()
#sull_score_cloglog_models = list()
#sull_score_cloglog_predictions = list()
# #Crowson, Atkinson, and & Therneau, 2016 approach
# #https://www.ncbi.nlm.nih.gov/pubmed/23907781
# linpred = list()
# linpred_grouping = list()
# predfit1 = list()
# predfit2 = list()
# predfit3 = list()
for(i in 1:ncol(sullscoremat)){
  sull_score_logit_models[[i]] = glm(test_ymatall[,i]~sullscoremat[,i], family=binomial(link="logit"))
  sull_score_logit_predictions[[i]] = predict(sull_score_logit_models[[i]], type="response")
  #sull_score_cloglog_models[[i]] = glm(test_ymatall[,i]~sullscoremat[,i], family=binomial(link="cloglog"))
  #sull_score_cloglog_predictions[[i]] = predict(sull_score_cloglog_models[[i]], type="response")
}

# #run summary statistics on preictions with test data
# for(i in 1:ncol(sullscoremat)){
#   #print(summary(sull_score_logit_models[[i]]))
#   #print(summary(sull_score_cloglog_models[[i]]))
#   print(cbind.data.frame(AIC(sull_score_logit_models[[i]]), AIC(sull_score_cloglog_models[[i]])))
# }

sull_score_logit_preds_dat = do.call("cbind.data.frame", sull_score_logit_predictions)
colnames(sull_score_logit_preds_dat) = c("lt1", "lt2", "lt3", "lt4", "lt5", "lt6", "gt6")

apply(sull_score_logit_preds_dat, 2, function(x) length(unique(x)))
apply(sull_score_logit_preds_dat, 2, function(x) data.frame(table(x)))


sullcat_hosmers = list()
sullcat_hosmer_predictions = list()
for(i in 1:ncol(test_ymatall)){
  #run hosmer and lemeshow groupings
  sullcat_hosmers[[i]] = grouped_calibration_function(
    TARGET=test_ymatall[,i], 
    #PREDICTION=sull_score_logit_predictions[[i]],
    PREDICTION=sullscoremat[,i],
    GROUPS_TO_CUT_BY=10, 
    P_VALUE_FOR_CONFIDENCE_INTERVAL=.05)
  sullcat_hosmers[[i]]$binned_obs_preds$sample = i
  #create datasets of predictions
  sullcat_hosmer_predictions[[i]] = sullcat_hosmers[[i]]$binned_obs_preds
}
sullcat_hosmer_preds_dat = do.call("rbind.data.frame", sullcat_hosmer_predictions)
sullcat_hosmer_preds_dat$sample = factor(sullcat_hosmer_preds_dat$sample)
head(sullcat_hosmer_preds_dat)

sullcat_hosmer_preds_dat$LOS = factor(as.character(mapvalues(sullcat_hosmer_preds_dat$sample,
                                                             from=c("1", "2", "3", "4", "5", "6", "7"),
                                                             to=c("a. <1 Month", "b. 1 to 2 Months", "c. 2 to 3 Months", "d. 3 to 4 Months",
                                                                  "e. 4 to 5 Months", "f. 5 to 6 Months", "g. 6+ Months"))))

#calibrations plot 
sullcalibsplot = ggplot(sullcat_hosmer_preds_dat) + 
  geom_abline(intercept=0, slope=1, size=.5) +
  #geom_hline(yintercept=0, colour="black", size=.3) +
  geom_line(aes(x=mean_predicted, y=mean_observed, group=LOS, colour=LOS), size=3) +
  #geom_errorbarh(aes(x=group_p_logit, y=mean_observed, xmin=group_ll_logit, xmax=group_ul_logit, group=LOS, colour=LOS), size=1) +
  geom_point(aes(x=mean_predicted, y=mean_observed, group=LOS, colour=LOS), size=1) +
  # geom_smooth(aes(x=mean_predicted, y=mean_observed, group=LOS, colour=LOS),
  #             #             #method = "lm", formula = y~bs(x, 5), se=FALSE, size=2.5) +
  #             method = "lm", se=FALSE, size=1.5) +
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

##############################################################
##############################################################
##############################################################
#test internal cross-validated model on external validation dataset
##############################################################
##############################################################
##############################################################

test_yout = as.numeric(test_ymatall %*% c(0:6))

#predict(cvl1mod, newx = test_xmatall, s="lambda.1se", type = "class")
#put on probability scale
testpreds_lambda1se_class = as.numeric(predict(cvl1mod, newx = test_xmatall, s="lambda.1se", type="class")) 
testpreds_feasible_class = as.numeric(predict(cvl1mod, newx = test_xmatall, s=penalization_lambda, type="class"))
testpreds_lambda1se = predict(cvl1mod, newx = test_xmatall, s="lambda.1se", type="response")
testpreds_feasible = predict(cvl1mod, newx = test_xmatall, s=penalization_lambda, type="response")


##############################################################
##############################################################
##############################################################
#model discriminations
##############################################################
##############################################################
##############################################################

citation("pROC")

#can't plot, but mean AUC is presented below (Hand and Till, 2001 modeling approach)
#David J. Hand and Robert J. Till (2001). A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems. Machine Learning 45(2), p. 171-186. DOI: 10.1023/A:1010920819831.
multroc_lambda1se_class = multiclass.roc(test_yout, testpreds_lambda1se_class)
multroc_lambda1se_class
multroc_feasible_class = multiclass.roc(test_yout, testpreds_feasible_class)
multroc_feasible_class
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
  numformat = function(val) { sub("^(-?)0.", "\\1.", sprintf("%.2f", val)) }
  rocdat$AUC_CI_formatted = paste0(numformat(roco$auc), "% (CI: ", numformat(as.numeric(roco$ci)[1]), "% to ", numformat(as.numeric(roco$ci)[3]),"%)") 
  rocdat$auc = as.numeric(roco$ci)[2]
  rocdat$aucll = as.numeric(roco$ci)[1]
  rocdat$aucul = as.numeric(roco$ci)[3]
  rocdat$Full_Description = paste0(rocdat$Description, " AUC: ", rocdat$AUC_CI_formatted)
  rocretlist = list(rocdat=rocdat, roc_object=roco)
  return(rocretlist)
}

rocker0_lambda1se = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,1], MODEL_PREDICTION=testpreds_lambda1se[,1,1], MODEL_DESCRIPTION_TEXT="a. <1 Month")
rocker1_lambda1se = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,2], MODEL_PREDICTION=testpreds_lambda1se[,2,1], MODEL_DESCRIPTION_TEXT="b. 1 to 2 Months")
rocker2_lambda1se = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,3], MODEL_PREDICTION=testpreds_lambda1se[,3,1], MODEL_DESCRIPTION_TEXT="c. 2 to 3 Months")
rocker3_lambda1se = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,4], MODEL_PREDICTION=testpreds_lambda1se[,4,1], MODEL_DESCRIPTION_TEXT="d. 3 to 4 Months")
rocker4_lambda1se = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,5], MODEL_PREDICTION=testpreds_lambda1se[,5,1], MODEL_DESCRIPTION_TEXT="e. 4 to 5 Months")
rocker5_lambda1se = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,6], MODEL_PREDICTION=testpreds_lambda1se[,6,1], MODEL_DESCRIPTION_TEXT="f. 5 to 6 Months")
rocker6_lambda1se = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,7], MODEL_PREDICTION=testpreds_lambda1se[,7,1], MODEL_DESCRIPTION_TEXT="g. 6+ Months")

rocdats_lambda1se = rbind.data.frame(rocker0_lambda1se$rocdat, rocker1_lambda1se$rocdat, 
                           rocker2_lambda1se$rocdat, rocker3_lambda1se$rocdat, 
                           rocker4_lambda1se$rocdat, rocker5_lambda1se$rocdat, 
                           rocker6_lambda1se$rocdat)

head(rocdats_lambda1se)
summary(rocdats)

rocker0_lambda1se$roc_object

discrimsplot_lambda1se = ggplot(rocdats_lambda1se) +
  geom_abline(intercept=0, slope=1, size=1) +
  geom_line(data=rocdats_lambda1se, aes(x=100-specificities, y=sensitivities, group=Full_Description, colour=Full_Description, linetype=Full_Description), size=1.25) +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  scale_linetype_discrete(name="Length of Stay") +
  scale_colour_manual(values=c("mediumorchid1", "mediumorchid3", "mediumpurple3", "purple1", "purple2", "purple3", "purple4"), name="Length of Stay") +
  #facet_wrap(~Full_Description, nrow=1) +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,1)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,1)) +
  scale_x_continuous(limits=c(0, 100)) +
  scale_y_continuous(limits=c(0, 100)) +
  ggtitle("Multinomial LASSO Best Cross-Validation Model External Validation Prediction Discrimination Values") +
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
discrimsplot_lambda1se
ggsave("LASSO_lambda1se_complete_feature_space_EPLOS_discrimination.pdf", discrimsplot_lambda1se, width=8, height=6, scale=1.2)


rocker0_feasible = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,1], MODEL_PREDICTION=testpreds_feasible[,1,1], MODEL_DESCRIPTION_TEXT="a. <1 Month")
rocker1_feasible = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,2], MODEL_PREDICTION=testpreds_feasible[,2,1], MODEL_DESCRIPTION_TEXT="b. 1 to 2 Months")
rocker2_feasible = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,3], MODEL_PREDICTION=testpreds_feasible[,3,1], MODEL_DESCRIPTION_TEXT="c. 2 to 3 Months")
rocker3_feasible = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,4], MODEL_PREDICTION=testpreds_feasible[,4,1], MODEL_DESCRIPTION_TEXT="d. 3 to 4 Months")
rocker4_feasible = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,5], MODEL_PREDICTION=testpreds_feasible[,5,1], MODEL_DESCRIPTION_TEXT="e. 4 to 5 Months")
rocker5_feasible = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,6], MODEL_PREDICTION=testpreds_feasible[,6,1], MODEL_DESCRIPTION_TEXT="f. 5 to 6 Months")
rocker6_feasible = discrimination_auroc_estimation(BINARY_TARGET=test_ymatall[,7], MODEL_PREDICTION=testpreds_feasible[,7,1], MODEL_DESCRIPTION_TEXT="g. 6+ Months")

rocdats_feasible = rbind.data.frame(rocker0_feasible$rocdat, rocker1_feasible$rocdat, 
                                     rocker2_feasible$rocdat, rocker3_feasible$rocdat, 
                                     rocker4_feasible$rocdat, rocker5_feasible$rocdat, 
                                     rocker6_feasible$rocdat)

head(rocdats_feasible)
summary(rocdats)

rocker0_feasible$roc_object

discrimsplot_feasible = ggplot(rocdats_feasible) +
  geom_abline(intercept=0, slope=1, size=1) +
  geom_line(data=rocdats_feasible, aes(x=100-specificities, y=sensitivities, group=Full_Description, colour=Full_Description, linetype=Full_Description), size=1.25) +
  #scale_colour_gradient2(low="firebrick", mid="grey70", high="navy", name="Importance") +
  scale_linetype_discrete(name="Length of Stay") +
  scale_colour_manual(values=c("lightblue", "dodgerblue1", "dodgerblue2", "dodgerblue", "dodgerblue3", "dodgerblue4", "navy"), name="Length of Stay") +
  #scale_colour_manual(values=c("greenyellow", "green", "springgreen1", "springgreen2", "springgreen3", "springgreen4", "darkgreen"), name="Length of Stay") +
  #facet_wrap(~Full_Description, nrow=1) +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,1)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,1)) +
  scale_x_continuous(limits=c(0, 100)) +
  scale_y_continuous(limits=c(0, 100)) +
  ggtitle("Multinomial LASSO Clinically Feasible Cross-Validation Model External Validation Prediction Discrimination Values") +
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
discrimsplot_feasible
ggsave("LASSO_feasible_complete_feature_space_EPLOS_discrimination.pdf", discrimsplot_feasible, width=8, height=6, scale=1.2)


#Note that sample sizes result in statistically significant differences even with very small effect size differences between classifiers
#This is the Delong Test to compare AUROCs
#test differences in AUCs between different learners
roc.test(rocker0_lambda1se$roc_object, rocker0_feasible$roc_object)
roc.test(rocker1_lambda1se$roc_object, rocker1_feasible$roc_object)
roc.test(rocker2_lambda1se$roc_object, rocker2_feasible$roc_object)
roc.test(rocker3_lambda1se$roc_object, rocker3_feasible$roc_object)
roc.test(rocker4_lambda1se$roc_object, rocker4_feasible$roc_object)
roc.test(rocker5_lambda1se$roc_object, rocker5_feasible$roc_object)
roc.test(rocker6_lambda1se$roc_object, rocker6_feasible$roc_object)

###################################################################
###################################################################
###################################################################
#Calibrations of classification for each category of the learner
###################################################################
###################################################################
###################################################################

grouped_calibration_function = function(TARGET, PREDICTION, SCALE=TRUE, GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05) {
  #require(Hmisc) #can't load Hmisc and dplyr, conflict, better to immediately call Hmisc functions without loading Hmisc package
  require(plyr)
  require(dplyr)
  ##Arguments:  ########################################################
  ##TARGET: target outcome (binary 0/1 variable). This is the original binary outcome to be predicted
  ##PREDICTION: predictions of each observation from the classifier. Should be real number
  ##GROUPS_T  qaO_CUT_BY: The number of bins to use in creating groups (a la Hosmer and Lemeshow)
  ##SCALE: Scale the pred to be between 0 and 1 before creating calibration? (logical, T/F)
  ##P_VALUE_FOR_CONFIDENCE_INTERVAL:  p value for confidence interval estimation 
  tarpred = cbind.data.frame(target = TARGET, predictor = PREDICTION) %>% arrange(predictor)
  minpredictor = min(tarpred$predictor)
  maxpredictor = max(tarpred$predictor)
  minmaxdiff = maxpredictor - minpredictor
  #scaling predictors if needed
  if (SCALE){
    tarpred$predictor = (tarpred$predictor - minpredictor)/minmaxdiff 
  }
  #tarpred$pred_group = cut(tarpred$predictor, breaks=GROUPS_TO_CUT_BY)
  tarpred$pred_group = .bincode(tarpred$predictor, 
                                breaks=quantile(tarpred$predictor, 
                                                probs=c(0, 1:GROUPS_TO_CUT_BY/GROUPS_TO_CUT_BY)
                                ), 
                                include.lowest=TRUE
  )
  binobs = tarpred %>% 
    group_by(pred_group) %>% 
    summarize(mean_observed = mean(target, na.rm=TRUE),
              mean_predicted = mean(predictor, na.rm=TRUE),
              sum_observed = sum(target, na.rm=TRUE),
              n_total = length(target)
    )
  
  ##Crowson Atkinson and Therneau 2016 Calibration
  ##https://www.ncbi.nlm.nih.gov/pubmed/23907781
  ##calibration slope estimation of current classifier prediction  
  #calib_in_the_large = glm(target ~ offset(predictor), data=tarpred, family=binomial(link="logit"))
  calibration_mod = glm(target~pred_group, data=tarpred, family=binomial(link="logit"))
  #hosmer_lemeshow_mod = glm(target~ -1 + pred_group + offset(predictor), data=tarpred, family=binomial(link="logit"))
  binobsgroupdat = data.frame(pred_group=unique(tarpred$pred_group))
  linpred_ests = predict(calibration_mod, newdata=binobsgroupdat, se.fit = TRUE)
  binobs$group_p_logit = boot::inv.logit(linpred_ests$fit)
  binobs$group_ll_logit = boot::inv.logit(linpred_ests$fit + linpred_ests$se*qnorm(P_VALUE_FOR_CONFIDENCE_INTERVAL/2))
  binobs$group_ul_logit = boot::inv.logit(linpred_ests$fit + linpred_ests$se*qnorm(1-(P_VALUE_FOR_CONFIDENCE_INTERVAL/2)))
  
  ##smoothed predictions (e.g., Copas et al. smoothing, but instead using more stable predictions for more variable predictions using mgcv algorithms)
  modgam = mgcv::gam(target~predictor, data=tarpred, family=binomial(link="logit"))
  predgams = predict(modgam, newdata=data.frame(predictor=binobs$mean_predicted), se.fit = TRUE)
  binobs$group_p_gam = boot::inv.logit(predgams$fit)
  binobs$group_ll_gam = boot::inv.logit(predgams$fit + predgams$se*qnorm(P_VALUE_FOR_CONFIDENCE_INTERVAL/2))
  binobs$group_ul_gam = boot::inv.logit(predgams$fit + predgams$se*qnorm(1-(P_VALUE_FOR_CONFIDENCE_INTERVAL/2)))
  
  ##possibly look in to isotonic approaches if issues arise with current methods
  ##https://www.analyticsvidhya.com/blog/2016/07/platt-scaling-isotonic-regression-minimize-logloss-error/
  
  ##Hosmer and Lemeshow groupings based on quantiles
  hltest = ResourceSelection::hoslem.test(tarpred$target, tarpred$predictor, g=GROUPS_TO_CUT_BY)
  HL_counts = data.frame(as.data.frame.matrix(hltest$observed), as.data.frame.matrix(hltest$expected), n = rowSums(as.data.frame.matrix(hltest$observed)))
  HLretlist = list(binned_obs_preds = binobs, HL_test = hltest, HL_counts = HL_counts)
  return(HLretlist)
}

########################################################################
#observed probability distribution for best cross-validation LASSO model
########################################################################
table(testdat$LOSmonths)

loscat_lasso_lambda1se = list()
loscat_lasso_lambda1se_predictions = list()
for(i in 1:ncol(test_ymatall)){
  #run hosmer and lemeshow groupings
  loscat_lasso_lambda1se[[i]] = grouped_calibration_function(TARGET=test_ymatall[,i], PREDICTION=testpreds_lambda1se[,i,1], 
                                                     GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05)
  loscat_lasso_lambda1se[[i]]$binned_obs_preds$sample = i
  #create datasets of predictions
  loscat_lasso_lambda1se_predictions[[i]] = loscat_lasso_lambda1se[[i]]$binned_obs_preds
}
loscat_lasso_lambda1se_preds_dat = do.call("rbind.data.frame", loscat_lasso_lambda1se_predictions)
loscat_lasso_lambda1se_preds_dat$sample = factor(loscat_lasso_lambda1se_preds_dat$sample)
head(loscat_lasso_lambda1se_preds_dat)

dput(levels(loscat_lasso_lambda1se_preds_dat$sample))
dput(levels(rocdats_lambda1se$Description))
loscat_lasso_lambda1se_preds_dat$LOS = factor(as.character(mapvalues(loscat_lasso_lambda1se_preds_dat$sample,
                                                            from=c("1", "2", "3", "4", "5", "6", "7"),
                                                            to=c("a. <1 Month", "b. 1 to 2 Months", "c. 2 to 3 Months", "d. 3 to 4 Months", 
                                                                 "e. 4 to 5 Months", "f. 5 to 6 Months", "g. 6+ Months"))))

head(loscat_lasso_lambda1se_preds_dat)

#calibrations plot 
calibsplot_lasso_lambda1se = ggplot(loscat_lasso_lambda1se_preds_dat) +
  geom_abline(intercept=0, slope=1, size=.5) +
  #geom_hline(yintercept=0, colour="black", size=.3) +
  geom_line(aes(x=mean_predicted, y=mean_observed, group=LOS, colour=LOS), size=3) +
  #geom_errorbarh(aes(x=group_p_logit, y=mean_observed, xmin=group_ll_logit, xmax=group_ul_logit, group=LOS, colour=LOS), size=1) +
  geom_point(aes(x=mean_predicted, y=mean_observed, group=LOS, colour=LOS), size=1) +
  # geom_smooth(aes(x=mean_predicted, y=mean_observed, group=LOS, colour=LOS),
  #             #             #method = "lm", formula = y~bs(x, 5), se=FALSE, size=2.5) +
  #             method = "lm", se=FALSE, size=1.5) +
  scale_colour_manual(values=c("mediumorchid1", "mediumorchid3", "mediumpurple3", "purple1", "purple2", "purple3", "purple4"), name="Length of Stay") +
  facet_wrap(~LOS, nrow=1) +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,0)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,0)) +
  scale_x_continuous(limits=c(0, 1), breaks = seq(0, 1, by=.2)) +
  scale_y_continuous(limits=c(0, 1), breaks = seq(0, 1, by=.1)) +
  ggtitle("Multinomial LASSO Best Cross-Validation Model External Validation Prediction Calibrations") +
  ylab("Observed Outcome Incidence") +
  xlab("Mean Prediction") +
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
        #strip.background = element_rect(fill="white"),
        legend.position = "none")
calibsplot_lasso_lambda1se
ggsave("LASSO_lambda1se_complete_feature_space_EPLOS_calibrations.pdf", calibsplot_lasso_lambda1se, width=11, height=6, scale=1.2)


########################################################################
#observed probability distribution for feasible cross-validation LASSO model
########################################################################
table(testdat$LOSmonths)

loscat_lasso_feasible = list()
loscat_lasso_feasible_predictions = list()
for(i in 1:ncol(test_ymatall)){
  #run hosmer and lemeshow groupings
  loscat_lasso_feasible[[i]] = grouped_calibration_function(TARGET=test_ymatall[,i], PREDICTION=testpreds_feasible[,i,1], 
                                                             GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05)
  loscat_lasso_feasible[[i]]$binned_obs_preds$sample = i
  #create datasets of predictions
  loscat_lasso_feasible_predictions[[i]] = loscat_lasso_feasible[[i]]$binned_obs_preds
}
loscat_lasso_feasible_preds_dat = do.call("rbind.data.frame", loscat_lasso_feasible_predictions)
loscat_lasso_feasible_preds_dat$sample = factor(loscat_lasso_feasible_preds_dat$sample)
head(loscat_lasso_feasible_preds_dat)

dput(levels(loscat_lasso_feasible_preds_dat$sample))
dput(levels(rocdats_feasible$Description))
loscat_lasso_feasible_preds_dat$LOS = factor(as.character(mapvalues(loscat_lasso_feasible_preds_dat$sample,
                                                                     from=c("1", "2", "3", "4", "5", "6", "7"),
                                                                     to=c("a. <1 Month", "b. 1 to 2 Months", "c. 2 to 3 Months", "d. 3 to 4 Months", 
                                                                          "e. 4 to 5 Months", "f. 5 to 6 Months", "g. 6+ Months"))))

head(loscat_lasso_feasible_preds_dat)

#calibrations plot 
calibsplot_lasso_feasible = ggplot(loscat_lasso_feasible_preds_dat) +
  geom_abline(intercept=0, slope=1, size=.5) +
  #geom_hline(yintercept=0, colour="black", size=.3) +
  geom_line(aes(x=mean_predicted, y=mean_observed, group=LOS, colour=LOS), size=3) +
  #geom_errorbarh(aes(x=group_p_logit, y=mean_observed, xmin=group_ll_logit, xmax=group_ul_logit, group=LOS, colour=LOS), size=1) +
  geom_point(aes(x=mean_predicted, y=mean_observed, group=LOS, colour=LOS), size=1) +
  # geom_smooth(aes(x=mean_predicted, y=mean_observed, group=LOS, colour=LOS),
  #             #             #method = "lm", formula = y~bs(x, 5), se=FALSE, size=2.5) +
  #             method = "lm", se=FALSE, size=1.5) +
  scale_colour_manual(values=c("lightblue", "dodgerblue1", "dodgerblue2", "dodgerblue", "dodgerblue3", "dodgerblue4", "navy"), name="Length of Stay") +
  facet_wrap(~LOS, nrow=1) +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,0)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,0)) +
  scale_x_continuous(limits=c(0, 1), breaks = seq(0, 1, by=.2)) +
  scale_y_continuous(limits=c(0, 1), breaks = seq(0, 1, by=.1)) +
  ggtitle("Multinomial LASSO Clinically Feasible Cross-Validation Model External Validation Prediction Calibrations") +
  ylab("Observed Outcome Incidence") +
  xlab("Mean Prediction") +
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
        #strip.background = element_rect(fill="white"),
        legend.position = "none")
calibsplot_lasso_feasible
ggsave("LASSO_feasible_complete_feature_space_EPLOS_calibrations.pdf", calibsplot_lasso_feasible, width=11, height=6, scale=1.2)

##############################################################
##############################################################
##############################################################
#script that produces the Tensorflow model
#keras_ann_full_phis_eplos_2016_2017_cohortV1.R
##############################################################
##############################################################
##############################################################

h5modload = keras::load_model_hdf5("EPLOS_sequential_ann_best_fit.h5", compile = TRUE)

bestmodel = h5modload

#########################################################################
#return predictions on external validation data (test dataset)
#########################################################################

bestpreds_classes = keras::predict_classes(bestmodel, test_xmatall)
bestpreds_probs = keras::predict_proba(bestmodel, test_xmatall)
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
multroc = multiclass.roc(testdat$LOSmonths, bestpreds_classes)
multroc
#likely quite low because some predictions from adjacent categories are close to each other since
#Lenght of Stay was discretized--so, some probabilities for some atients may be
#slightly higher for adjacent groups than the actual group fo the test data. 
#Example, the true group is 2 Month stay (e.g., 62 days) has the predicted probability of .1, while 
#the predicted probability of a 1 months stay while the predicted probabiliy of a 1 month stay (i.e., up to 59 days) is .11, so it 
#is predicted as a 1 month vs. a 2 months stay (true length of stay is 62 days)

#perfect description of interpretation of discrimination underfitting/overfitting and under-estimation vs over-estimation on pg 30
#ftp://ftp.esat.kuleuven.be/pub/SISTA/ida/reports/14-224.pdf 


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


#observed probability distribution
table(testdat$LOSmonths)

loscat_hosmers = list()
loscat_hosmer_predictions = list()
for(i in 1:ncol(test_ymatall)){
  #run hosmer and lemeshow groupings
  loscat_hosmers[[i]] = grouped_calibration_function(TARGET=test_ymatall[,i], PREDICTION=bestpreds_probs[,i], 
                                                     GROUPS_TO_CUT_BY=10, P_VALUE_FOR_CONFIDENCE_INTERVAL=.05)
  loscat_hosmers[[i]]$binned_obs_preds$sample = i
  #create datasets of predictions
  loscat_hosmer_predictions[[i]] = loscat_hosmers[[i]]$binned_obs_preds
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

head(loscat_hosmer_preds_dat)

#calibrations plot 
calibsplot = ggplot(loscat_hosmer_preds_dat) +
  geom_abline(intercept=0, slope=1, size=.5) +
  #geom_hline(yintercept=0, colour="black", size=.3) +
  geom_line(aes(x=mean_predicted, y=mean_observed, group=LOS, colour=LOS), size=3) +
  #geom_errorbarh(aes(x=group_p_logit, y=mean_observed, xmin=group_ll_logit, xmax=group_ul_logit, group=LOS, colour=LOS), size=1) +
  geom_point(aes(x=mean_predicted, y=mean_observed, group=LOS, colour=LOS), size=1) +
  # geom_smooth(aes(x=mean_predicted, y=mean_observed, group=LOS, colour=LOS),
  #             #             #method = "lm", formula = y~bs(x, 5), se=FALSE, size=2.5) +
  #             method = "lm", se=FALSE, size=1.5) +
  scale_colour_manual(values=c("gold2", "orange", "orangered", "red", "firebrick1", "firebrick", "firebrick4"), name="Model") +
  facet_wrap(~LOS, nrow=1) +
  #scale_x_continuous(limits=c(0, 1), expand = c(0,0)) +
  #scale_y_continuous(limits=c(0, 1), expand = c(0,0)) +
  scale_x_continuous(limits=c(0, 1), breaks = seq(0, 1, by=.2)) +
  scale_y_continuous(limits=c(0, 1), breaks = seq(0, 1, by=.1)) +
  ggtitle("Full ANN Model Prediction") +
  ylab("Observed Outcome Incidence") +
  xlab("Mean Prediction") +
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
        #strip.background = element_rect(fill="white"),
        legend.position = "none")
calibsplot
ggsave("complete_feature_space_EPLOS_calibration.pdf", calibsplot, width=11, height=6, scale=1.2)


#########################################################################
#########################################################################
#########################################################################
#combine discrimination plots
#########################################################################
#########################################################################
#########################################################################

#ardiscrimplots = gridExtra::grid.arrange(discrimsplot, discrimsplot_lambda1se, discrimsplot_feasible, ncol=1)
ardiscrimplots = gridExtra::grid.arrange(discrimsplot, discrimsplot_lambda1se, sulldiscrimsplot, ncol=1)
ardiscrimplots
ggsave("Figure1_disciminations_EPLOS_combined.pdf", ardiscrimplots, width=11, height=8, scale=1.2)


#arcalibsplots = gridExtra::grid.arrange(calibsplot, calibsplot_lasso_lambda1se, calibsplot_lasso_feasible, ncol=1)
arcalibsplots = gridExtra::grid.arrange(calibsplot, calibsplot_lasso_lambda1se, sullcalibsplot, ncol=1)
arcalibsplots
ggsave("Figure2_calibrations_EPLOS_discrimination.pdf", arcalibsplots, width=8, height=11, scale=1.5)


###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
#simulate observed population dynames for scores
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################

all.equal(dimnames(xmatall)[[2]], dimnames(test_xmatall)[[2]])
ncol(xmatall)
ncol(test_xmatall)

fulldat = rbind.data.frame(xmatall, test_xmatall)
ncol(fulldat)
length(dimnames(fulldat)[[2]])


###################################################################
#phi correlations
###################################################################

phis_phis = Matrix::nearPD(round(as.matrix(var(fulldat)), digits=2))
phis_props = apply(fulldat, 2, function(x) mean(x, na.rm=TRUE))

outcome_names = c("< 1 Month", "1-2 Months", "2-3 Months", "3-4 Months", "4-5 Months", "5-6 Months", "6 Months or more")

ITERATIONS = 100

###################################################################
#for full ANN scoring algorithm
###################################################################

markov_binary_ANN_full_model_sim_scoring = function(SIMULATED_CASES, MARG_PROBS, MARG_COVAR, MODEL, QUANTILES, COLNAMES){
  probmat = rmvbin(SIMULATED_CASES, margprob=MARG_PROBS)
  modpreds = data.frame(keras::predict_proba(MODEL, probmat)) #model predictions from Keras 
  colnames(modpreds) = COLNAMES
  modquants = cbind.data.frame(sapply(modpreds, function(x) data.frame(quantile(x, probs=c(0, 1:QUANTILES/QUANTILES)))))
  colnames(modquants) = COLNAMES
  modquants$quantile = paste0(c(0, 1:QUANTILES/QUANTILES)*100, "%")
  return(modquants)
}

###############################################
#make sure to load model to predict based on ANN model
###############################################
#h5modload = keras::load_model_hdf5("EPLOS_sequential_ann_best_fit.h5", compile = TRUE)
#bestmodel = h5modload

markov_ann_prob_sims = list()
for(i in 1:ITERATIONS){
  for(j in 1:length(outcome_names)){
    markov_ann_prob_sims[[i]] = markov_binary_ANN_full_model_sim_scoring(SIMULATED_CASES=10000,
                                                                         MARG_PROBS=phis_props, 
                                                                         MARG_COVAR=phis_phis$mat,
                                                                         MODEL = bestmodel,
                                                                         QUANTILES=20,
                                                                         COLNAMES = outcome_names)
    markov_ann_prob_sims[[i]]$order = 1:nrow(markov_ann_prob_sims[[i]])
    markov_ann_prob_sims[[i]]$iteration = i
    print(paste0("########################### Running Iteration ", i, " ###########################"))
  }
}

ann_markov_sims_stacked = reshape2::melt(
  do.call("rbind.data.frame", markov_ann_prob_sims), id=c("quantile", "order", "iteration")) %>%
  group_by(variable, quantile, order) %>%
  summarize(n = n(), 
            mean_score = mean(value, na.rm=TRUE),
            sd_score = sd(value, na.rm=TRUE)) %>%
  arrange(variable, order)

colnames(ann_markov_sims_stacked) = c("description", "quantile", "order", "n", "mean_score", "sd_score")
head(ann_markov_sims_stacked)

markov_annsub = subset(ann_markov_sims_stacked, description!="< 1 Month")

percannplot = ggplot(ann_markov_sims_stacked) +
  #geom_hline(yintercept=0, colour="black", size=.3) +
  geom_line(aes(x=reorder(quantile, order), y=mean_score, group=description, colour=description, linetype=description), size=1.5) +
  #geom_errorbarh(aes(x=group_p_logit, y=mean_observed, xmin=group_ll_logit, xmax=group_ul_logit, group=LOS, colour=LOS), size=1) +
  #geom_point(aes(x=reorder(quantile, order), y=mean_score, group=description, colour=description)) +
  # geom_smooth(aes(x=reorder(quantile, order), y=mean_score, group=description, colour=description),
  #              method = "gam", formula = y~bs(x, 6), se=FALSE, size=1) +
  scale_linetype_discrete(name="Length of Stay") +
  scale_colour_manual(values=c("gold2", "orange", "orangered", "red", "firebrick1", "firebrick", "firebrick4"), name="Length of Stay") +
  #scale_colour_manual(name="Length of Stay", values=c("orange", "orangered", "red", "firebrick1", "firebrick", "firebrick4")) +
  #facet_wrap(~description, nrow=1) +
  #scale_x_continuous(limits=c(0, 100)) +
  scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, by=.1)) +
  ggtitle("Estimated Quantiles of Risk Distribution:\nFull Artificial Neural Network Model Predictions") +
  ylab("Model Predicted Risk Score") +
  xlab("Percentile of Risk") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        panel.grid.major=element_line(colour="grey90", size=.2),
        #panel.grid.minor = element_blank(),
        legend.position = "right")
percannplot
ggsave("full_ANN_markov_percentiles.pdf", percannplot, width=8, height=6, scale=1.2)

###################################################################
#markov estimation of best fitting LASSO multinomial lambda model
###################################################################

#"lambda.1se"
#penalization_lambda

markov_glmnet_sim_predscoring = function(SIMULATED_CASES, 
                                         MARG_PROBS, 
                                         MARG_COVAR,
                                         MODEL,
                                         LAMBDA, 
                                         QUANTILES,
                                         PRED_COLUMN_NAMES){
  #null vcov estimates uncorrelated draws; tenable when infrequent outcomes and vcov that may not always lead to semi-positive definite matrices
  binmat = rmvbin(SIMULATED_CASES, margprob=MARG_PROBS)
  scoremat = predict(MODEL, newx=binmat, s=LAMBDA, type="response")
  quantmat = data.frame(apply(scoremat, 2, function(x) quantile(x, probs=c(0, 1:QUANTILES/QUANTILES))))
  colnames(quantmat) = PRED_COLUMN_NAMES
  quantmat$quantile = dimnames(quantmat)[[1]]
  quantmat$order = 1:nrow(quantmat)
  #retlist = list(descriptives=descvec, quantiles=quantmat) 
  #return(retlist)
  return(quantmat)
}

################################################################
#best markov simulations using fitting internally cross-validated LASSO model
################################################################

bestlassosims = list()
for(i in 1:ITERATIONS){
  bestlassosims[[i]] = markov_glmnet_sim_predscoring(SIMULATED_CASES=10000,
                                             MARG_PROBS=phis_props, 
                                             MARG_COVAR=phis_phis$mat,
                                             MODEL=cvl1mod,
                                             LAMBDA="lambda.1se",
                                             QUANTILES=20,
                                             PRED_COLUMN_NAMES=outcome_names)
  bestlassosims[[i]]$iteration = i
  print(paste0("########################### Running Iteration ", i, " ###########################"))
}

head(bestlassosims)

bestlassosimsdat = reshape2::melt(do.call("rbind.data.frame", bestlassosims), id=c("quantile", "order", "iteration")) %>%
  group_by(quantile, order, variable) %>%
  summarize(n = n(), 
            mean_score = mean(value, na.rm=TRUE),
            sd_score = sd(value, na.rm=TRUE)) %>%
  arrange(order)

bestlassosimsdat
colnames(bestlassosimsdat) = c("quantile", "order", "description", "n", "mean_score", "sd_score")

percbestlassoplot = ggplot(bestlassosimsdat) +
  #geom_hline(yintercept=0, colour="black", size=.3) +
  geom_line(aes(x=reorder(quantile, order), y=mean_score, group=description, colour=description, linetype=description), size=1.5) +
  #geom_errorbarh(aes(x=group_p_logit, y=mean_observed, xmin=group_ll_logit, xmax=group_ul_logit, group=LOS, colour=LOS), size=1) +
  #geom_point(aes(x=reorder(quantile, order), y=mean_score, group=description, colour=description)) +
  # geom_smooth(aes(x=reorder(quantile, order), y=mean_score, group=description, colour=description),
  #              method = "gam", formula = y~bs(x, 6), se=FALSE, size=1) +
  scale_colour_manual(values=c("mediumorchid1", "mediumorchid3", "mediumpurple3", "purple1", "purple2", "purple3", "purple4"), name="Length of Stay") +
  #scale_colour_discrete(name="Length of Stay") + 
  scale_linetype_discrete(name="Length of Stay") +
  #facet_wrap(~description, nrow=1) +
  #scale_x_continuous(limits=c(0, 100)) +
  scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, by=.1)) +
  ggtitle("Estimated Quantiles of Risk Distribution:\nBest Multinomial LASSO Model") +
  ylab("Model Predicted Risk Score") +
  xlab("Percentile of Risk") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        panel.grid.major=element_line(colour="grey90", size=.2),
        #panel.grid.minor = element_blank(),
        legend.position = "right")
percbestlassoplot
ggsave("best_LASSO_multinomial_markov_percentiles.pdf", percbestlassoplot, width=8, height=6, scale=1.2)

################################################################
#feasibile markov simulations using fitting internally cross-validated LASSO model
################################################################

feaslassosims = list()
for(i in 1:ITERATIONS){
  feaslassosims[[i]] = markov_glmnet_sim_predscoring(SIMULATED_CASES=10000,
                                                     MARG_PROBS=phis_props, 
                                                     MARG_COVAR=phis_phis$mat,
                                                     MODEL=cvl1mod,
                                                     LAMBDA=penalization_lambda,
                                                     QUANTILES=20,
                                                     PRED_COLUMN_NAMES=outcome_names)
  feaslassosims[[i]]$iteration = i
  print(paste0("########################### Running Iteration ", i, " ###########################"))
}

feaslassosimsdat = reshape2::melt(do.call("rbind.data.frame", feaslassosims), id=c("quantile", "order", "iteration")) %>%
  group_by(quantile, order, variable) %>%
  summarize(n = n(), 
            mean_score = mean(value, na.rm=TRUE),
            sd_score = sd(value, na.rm=TRUE)) %>%
  arrange(order)

feaslassosimsdat
colnames(feaslassosimsdat) = c("quantile", "order", "description", "n", "mean_score", "sd_score")

percfeaslassoplot = ggplot(feaslassosimsdat) +
  #geom_hline(yintercept=0, colour="black", size=.3) +
  geom_line(aes(x=reorder(quantile, order), y=mean_score, group=description, colour=description, linetype=description), size=1.5) +
  #geom_errorbarh(aes(x=group_p_logit, y=mean_observed, xmin=group_ll_logit, xmax=group_ul_logit, group=LOS, colour=LOS), size=1) +
  #geom_point(aes(x=reorder(quantile, order), y=mean_score, group=description, colour=description)) +
  # geom_smooth(aes(x=reorder(quantile, order), y=mean_score, group=description, colour=description),
  #              method = "gam", formula = y~bs(x, 6), se=FALSE, size=1) +
  #scale_colour_manual(values=c("mediumorchid1", "mediumorchid3", "mediumpurple3", "purple1", "purple2", "purple3", "purple4"), name="Length of Stay") +
  scale_colour_manual(values=c("lightblue", "dodgerblue1", "dodgerblue2", "dodgerblue", "dodgerblue3", "dodgerblue4", "navy"), name="Length of Stay") +
  #scale_colour_discrete(name="Length of Stay") + 
  scale_linetype_discrete(name="Length of Stay") +
  #facet_wrap(~description, nrow=1) +
  #scale_x_continuous(limits=c(0, 100)) +
  scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, by=.1)) +
  ggtitle("Estimated Quantiles of Risk Distribution:\nFeasible By-Hand Scoring of LASSO Model") +
  ylab("Model Predicted Risk Score") +
  xlab("Percentile of Risk") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        panel.grid.major=element_line(colour="grey90", size=.2),
        #panel.grid.minor = element_blank(),
        legend.position = "right")
percfeaslassoplot
ggsave("feasbile_LASSO_multinomial_markov_percentiles.pdf", percfeaslassoplot, width=8, height=6, scale=1.2)


####################################################################
####################################################################
####################################################################
#markov modeling of sullivan scoring
####################################################################
####################################################################
####################################################################

#develop a point scoring vector for each outcome category based on selected features' points only (7 in total)
featlistdat = data.frame(feature = dimnames(fulldat)[[2]])

fullss0points = merge(featlistdat, ss0, by=c("feature"), all.x=TRUE)
fullss0points[is.na(fullss0points)] = 0

fullss1points = merge(featlistdat, ss1, by=c("feature"), all.x=TRUE)
fullss1points[is.na(fullss1points)] = 0

fullss2points = merge(featlistdat, ss2, by=c("feature"), all.x=TRUE)
fullss2points[is.na(fullss2points)] = 0

fullss3points = merge(featlistdat, ss3, by=c("feature"), all.x=TRUE)
fullss3points[is.na(fullss3points)] = 0

fullss4points = merge(featlistdat, ss4, by=c("feature"), all.x=TRUE)
fullss4points[is.na(fullss4points)] = 0

fullss5points = merge(featlistdat, ss5, by=c("feature"), all.x=TRUE)
fullss5points[is.na(fullss5points)] = 0

fullss6points = merge(featlistdat, ss6, by=c("feature"), all.x=TRUE)
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


apply(phis_sull_scores, 2, function(x) quantile(x, probs = seq(0, 1, by=.1)))


###################################################################
#markov estimation simplified sullivan scoring algorithm
###################################################################

markov_binary_sim_scoring = function(SIMULATED_CASES, MARG_PROBS, MARG_COVAR, SCORING_VECTOR, QUANTILES, DESCRIPTION_TEXT){
  #null vcov estimates uncorrelated draws; tenable when infrequent outcomes and vcov that may not always lead to semi-positive definite matrices
  scoremat = rmvbin(SIMULATED_CASES, margprob=MARG_PROBS) %*% SCORING_VECTOR
  #descvec = psych::describe(scoremat)
  quantmat = data.frame(quantile(scoremat, probs=c(0, 1:QUANTILES/QUANTILES)))
  quantmat$quantiles = dimnames(quantmat)[[1]]
  colnames(quantmat) = c("score", "quantile")
  quantmat$description = DESCRIPTION_TEXT
  quantmat$order = 1:nrow(quantmat)
  #retlist = list(descriptives=descvec, quantiles=quantmat) 
  #return(retlist)
  return(quantmat)
}


######################################################################
#quantile for less than 1 month
######################################################################

percsims0 = list()
for(i in 1:ITERATIONS){
  percsims0[[i]] = markov_binary_sim_scoring(SIMULATED_CASES=10000, 
                                             MARG_PROBS=phis_props, 
                                             MARG_COVAR=phis_phis$mat,
                                             SCORING_VECTOR=fullss0points$sullivan_points, 
                                             QUANTILES=20,
                                             DESCRIPTION_TEXT = "< 1 Month")
  percsims0[[i]]$iteration = i
  percsims0
}

percsims0dat = do.call("rbind.data.frame", percsims0) %>%
  group_by(description, quantile, order) %>%
  summarize(n = n(), 
            mean_score = mean(score, na.rm=TRUE),
            sd_score = sd(score, na.rm=TRUE)) %>%
  arrange(order)

percsims0dat

######################################################################
#quantile for 1 to 2 months
######################################################################

percsims1 = list()
for(i in 1:ITERATIONS){
  percsims1[[i]] = markov_binary_sim_scoring(SIMULATED_CASES=10000, 
                                             MARG_PROBS=phis_props, 
                                             MARG_COVAR=phis_phis$mat,
                                             SCORING_VECTOR=fullss1points$sullivan_points, 
                                             QUANTILES=20,
                                             DESCRIPTION_TEXT = "1-2 Months")
  percsims1[[i]]$iteration = i
  percsims1
}

percsims1dat = do.call("rbind.data.frame", percsims1) %>%
  group_by(description, quantile, order) %>%
  summarize(n = n(), 
            mean_score = mean(score, na.rm=TRUE),
            sd_score = sd(score, na.rm=TRUE)) %>%
  arrange(order)

percsims1dat

######################################################################
#quantile for 2 to 3 months
######################################################################

percsims2 = list()
for(i in 1:ITERATIONS){
  percsims2[[i]] = markov_binary_sim_scoring(SIMULATED_CASES=10000, 
                                             MARG_PROBS=phis_props, 
                                             MARG_COVAR=phis_phis$mat,
                                             SCORING_VECTOR=fullss2points$sullivan_points, 
                                             QUANTILES=20,
                                             DESCRIPTION_TEXT = "2-3 Months")
  percsims2[[i]]$iteration = i
  percsims2
}

percsims2dat = do.call("rbind.data.frame", percsims2) %>%
  group_by(description, quantile, order) %>%
  summarize(n = n(), 
            mean_score = mean(score, na.rm=TRUE),
            sd_score = sd(score, na.rm=TRUE)) %>%
  arrange(order)

percsims2dat

######################################################################
#quantile for 3 to 4 months
######################################################################

percsims3 = list()
for(i in 1:ITERATIONS){
  percsims3[[i]] = markov_binary_sim_scoring(SIMULATED_CASES=10000, 
                                             MARG_PROBS=phis_props, 
                                             MARG_COVAR=phis_phis$mat,
                                             SCORING_VECTOR=fullss3points$sullivan_points, 
                                             QUANTILES=20,
                                             DESCRIPTION_TEXT = "3-4 Months")
  percsims3[[i]]$iteration = i
  percsims3
}

percsims3dat = do.call("rbind.data.frame", percsims3) %>%
  group_by(description, quantile, order) %>%
  summarize(n = n(), 
            mean_score = mean(score, na.rm=TRUE),
            sd_score = sd(score, na.rm=TRUE)) %>%
  arrange(order)

percsims3dat


######################################################################
#quantile for 4 to 5 months
######################################################################

percsims4 = list()
for(i in 1:ITERATIONS){
  percsims4[[i]] = markov_binary_sim_scoring(SIMULATED_CASES=10000, 
                                             MARG_PROBS=phis_props, 
                                             MARG_COVAR=phis_phis$mat,
                                             SCORING_VECTOR=fullss4points$sullivan_points, 
                                             QUANTILES=20,
                                             DESCRIPTION_TEXT = "4-5 Months")
  percsims4[[i]]$iteration = i
  percsims4
}

percsims4dat = do.call("rbind.data.frame", percsims4) %>%
  group_by(description, quantile, order) %>%
  summarize(n = n(), 
            mean_score = mean(score, na.rm=TRUE),
            sd_score = sd(score, na.rm=TRUE)) %>%
  arrange(order)

percsims4dat

######################################################################
#quantile for 5 to 6 months
######################################################################

percsims5 = list()
for(i in 1:ITERATIONS){
  percsims5[[i]] = markov_binary_sim_scoring(SIMULATED_CASES=10000, 
                                             MARG_PROBS=phis_props, 
                                             MARG_COVAR=phis_phis$mat,
                                             SCORING_VECTOR=fullss5points$sullivan_points, 
                                             QUANTILES=20,
                                             DESCRIPTION_TEXT = "5-6 Months")
  percsims5[[i]]$iteration = i
  percsims5
}

percsims5dat = do.call("rbind.data.frame", percsims5) %>%
  group_by(description, quantile, order) %>%
  summarize(n = n(), 
            mean_score = mean(score, na.rm=TRUE),
            sd_score = sd(score, na.rm=TRUE)) %>%
  arrange(order)

percsims5dat

######################################################################
#quantile for 6+ months
######################################################################

percsims6 = list()
for(i in 1:ITERATIONS){
  percsims6[[i]] = markov_binary_sim_scoring(SIMULATED_CASES=10000, 
                                             MARG_PROBS=phis_props, 
                                             MARG_COVAR=phis_phis$mat,
                                             SCORING_VECTOR=fullss6points$sullivan_points, 
                                             QUANTILES=20,
                                             DESCRIPTION_TEXT = "6 Months or more")
  percsims6[[i]]$iteration = i
  percsims6
}

percsims6dat = do.call("rbind.data.frame", percsims6) %>%
  group_by(description, quantile, order) %>%
  summarize(n = n(), 
            mean_score = mean(score, na.rm=TRUE),
            sd_score = sd(score, na.rm=TRUE)) %>%
  arrange(order)

percsims6dat


percsimsall = rbind.data.frame(
  #                      percsims0dat,
  percsims1dat, 
  percsims2dat, 
  percsims3dat, 
  percsims4dat, 
  percsims5dat, 
  percsims6dat)

#percsimsall$se = percsimsall$sd_score/sqrt(percsimsall$n)
#percsimsall$mean_score = round(percsimsall$mean_score, digits=0)

percsullplot = ggplot(percsimsall) +
  #geom_hline(yintercept=0, colour="black", size=.3) +
  geom_line(aes(x=reorder(quantile, order), y=mean_score, group=description, colour=description, linetype=description), size=1.5) +
  #geom_errorbarh(aes(x=group_p_logit, y=mean_observed, xmin=group_ll_logit, xmax=group_ul_logit, group=LOS, colour=LOS), size=1) +
  #geom_point(aes(x=reorder(quantile, order), y=mean_score, group=description, colour=description)) +
  # geom_smooth(aes(x=reorder(quantile, order), y=mean_score, group=description, colour=description),
  #              method = "gam", formula = y~bs(x, 6), se=FALSE, size=1) +
  #scale_colour_manual(values=c("lightblue", "dodgerblue1", "dodgerblue2", "dodgerblue", "dodgerblue3", "dodgerblue4", "navy"), name="Length of Stay") +
  scale_colour_manual(name="Length of Stay", values=c("dodgerblue1", "dodgerblue2", "dodgerblue", "dodgerblue3", "dodgerblue4", "navy")) +
  scale_linetype_discrete(name="Length of Stay") +
  #facet_wrap(~description, nrow=1) +
  #scale_x_continuous(limits=c(0, 100)) +
  #scale_y_continuous(limits=c(0, 100)) +
  ggtitle("Estimated Quantiles of Risk Distribution:\nSimplified Sullivan Point Scoring System") +
  ylab("Simplified Point Score") +
  xlab("Percentile of Risk") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        panel.grid.major=element_line(colour="grey90", size=.2),
        #panel.grid.minor = element_blank(),
        legend.position = "right")
percsullplot
ggsave("simplified_sullivan_markov_percentiles.pdf", percsullplot, width=8, height=6, scale=1.2)



##########################################################
#quantile plots combined
##########################################################

#markov_quantplots = gridExtra::grid.arrange(percannplot, percbestlassoplot, percfeaslassoplot, ncol=1)
markov_quantplots = gridExtra::grid.arrange(percannplot, percbestlassoplot, percsullplot, ncol=1)
markov_quantplots
ggsave("Figure3_markov_quantile_plots_arranged.pdf", markov_quantplots, width=8, height=8, scale=1.5)


##############################################################
##############################################################
##############################################################
#SAVE IMAGE
##############################################################
##############################################################
##############################################################

#save.image("~/MCHS/ELOS/PHIS/EPLOS_multinomial_grouped_lasso_predictions.RData")

