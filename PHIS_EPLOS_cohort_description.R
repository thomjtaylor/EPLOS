

set.seed(57006)

rm(list=ls())

library(plyr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(scales)
library(survey)
library(DescTools)

#library(extrafont)
#guidance on font embedding (Note that if requiring Arial, default Helvetica is virtually the same in ggplot2)
#https://github.com/wch/extrafont 
# loadfonts(device="win")
# Sys.setenv(R_GSCMD = "C:/Program Files/gs/gs9.23/bin/gswin64c.exe")
# embed_fonts("font_ggplot.pdf", outfile="font_ggplot_embed.pdf")

setwd("C:/Users/Owner/Documents/MCHS/ELOS/PHIS/")

###################################################
#script
###################################################
#/home/thomas.taylor/Totapally_PICU_KID_EPLOS/R/PHIS_EPLOS_cohort_description.R

###################################################
#load data
###################################################
#load("~/MCHS/ELOS/PHIS/PHIS_EPLOS_cohort_description.RData")

# tempall = read.csv("phis_eplos_2016_2017_cohort.csv")[,-1]
# temp = tempall[tempall$age_yrs<=.08333333,]
# rm(tempall)

format(object.size(temp), units = "auto")

glimpse(temp[,1:35])

str(temp$Discharge_Date)
temp$Discharge_Date = as.character(temp$Discharge_Date)
temp$Discharge_Date = ymd(temp$Discharge_Date)

#outcome categorization for demographic data contrasts
temp$LOS_category = NA
temp$LOS_category[temp$LOS<=30] = "a. <1 Month"
temp$LOS_category[temp$LOS>30 & temp$LOS<=60] = "b. 1 to 2 Months"
temp$LOS_category[temp$LOS>60 & temp$LOS<=90] = "c. 2 to 3 Months"
temp$LOS_category[temp$LOS>90 & temp$LOS<=120] = "d. 3 to 4 Months"
temp$LOS_category[temp$LOS>120 & temp$LOS<=150] = "e. 4 to 5 Months"
temp$LOS_category[temp$LOS>150 & temp$LOS<=180] = "f. 5 to 6 Months"
temp$LOS_category[temp$LOS>180] = "g. 6+ Months"
temp$LOS_category = factor(temp$LOS_category)
table(temp$LOS_category)

table(temp$LOS_category, temp$LOSmonths)

summary(temp$age_yrs)
temp$age_days = temp$age_yrs*365
summary(temp$age_days)

####################################################
#subset data to only variables of interest in analyses for manuscript (e.g., features and demographics)
####################################################
dput(names(temp))

descriptive_vars_to_include = c("LOS_category", 
                         "Patient_Type_Title", "age_days",
                         "Gender_Title", "Ethnicity_Title", "Race.White", "Race.Black", 
                         "Race.Asian", "Race.Pacific_Islander", "Race.American_Indian", "Race.Other", 
                         "admit_type", "insurance", "died_in_hospital",
                         grep("DX_", names(temp), value=TRUE),
                         grep("PX_", names(temp), value=TRUE))
descriptive_vars_to_include[1:20]

demdat = subset(temp, select=descriptive_vars_to_include)
rm(temp)

#change all variables in to factor to run through survey svyby function without issues in concatenation in formatting function
demdat = demdat %>% 
  mutate_if(is.numeric, as.factor) %>%
  mutate_if(is.integer, as.factor)

demdat$age_days = as.numeric(as.character(demdat$age_days))
glimpse(demdat)

####################################################
#FUNCTIONS
####################################################

numformat = function(val) { sub("^(-?)0.", "\\1.", sprintf("%.2f", val)) }

quantile_cut_function=function(VARIABLE_NAME, N_QUANTILES) {
  cut(VARIABLE_NAME, breaks=c(quantile(VARIABLE_NAME, probs = seq(0, 1, by = (1/N_QUANTILES)))), include.lowest=TRUE)
  #labels=c("0-20","20-40","40-60","60-80","80-100"), include.lowest=TRUE)
}

svy_single_variable_formatter = function(DESIGN, ROW_VAR, ROW_VAR_TEXT_DESCRIPTION, PRINT_DIGITS=4){
  print(paste0("########## Currently Running: ", ROW_VAR, " ##########"))
  formrow = as.formula(paste0("~", ROW_VAR))
  longtots = cbind.data.frame(svytotal(formrow, design=DESIGN, keep.var=FALSE), round(confint(svytotal(formrow, design=DESIGN), digits=0)))
  colnames(longtots) = c("n", "se_n", "ll", "ul")
  longtots$n = round(longtots$n, digits=0)
  longtots$ll = ifelse(longtots$ll<0, 0, longtots$ll)
  longprops = cbind.data.frame(svymean(formrow, design=DESIGN, keep.var=FALSE), round(confint(svymean(formrow, design=DESIGN)), digits=PRINT_DIGITS))
  colnames(longprops) = c("prop", "se_prop", "ll", "ul")
  longprops$prop = round(longprops$prop, digits=PRINT_DIGITS)
  longprops$ll = round(longprops$ll, digits=PRINT_DIGITS)
  longprops$ll = ifelse(longprops$ll<0, 0, longprops$ll)
  longprops$ul = round(longprops$ul, digits=PRINT_DIGITS)
  ###############################
  #after wide form datasets created, add in formatted column for long form
  ###############################
  longtots$formattedN = paste0(prettyNum(longtots$n, big.mark=","), " (", prettyNum(longtots$ll, big.mark=","), " - ", prettyNum(longtots$ul, big.mark=","), ")")
  #number formatter #################################
  numformat = function(val) { sub("^(-?)0.", "\\1.", sprintf("%.2f", val)) }
  longprops$formattedP = paste0(numformat(longprops$prop*100), "% (", numformat(longprops$ll*100), "% - ", numformat(longprops$ul*100), "%)")
  longtots1 = cbind.data.frame(varname = rep(ROW_VAR, times=nrow(longtots)),
                               Row_Variable = rep(ROW_VAR_TEXT_DESCRIPTION, times=nrow(longtots)), 
                               level = sub(ROW_VAR, "", dimnames(longtots)[[1]]),
                               longtots)
  longprops1 = cbind.data.frame(varname = rep(ROW_VAR, times=nrow(longprops)),
                                Row_Variable = rep(ROW_VAR_TEXT_DESCRIPTION, times=nrow(longprops)), 
                                level = sub(ROW_VAR, "", dimnames(longprops)[[1]]),
                                longprops)
  #simple N and proportions dataset with formatting
  simpledat = cbind.data.frame(subset(longtots1, select=c(varname, Row_Variable, level, n)), subset(longprops1, select=c(prop)))
  simpledat$formatted = paste0(prettyNum(simpledat$n, big.mark=","), " (", numformat(simpledat$prop*100), "%)")
  simpledat$formatted_n = prettyNum(simpledat$n, big.mark=",")
  simpledat$formatted_prop = paste0(numformat(simpledat$prop*100), "%")
  #return datasets
  returnlist = list(long_totals = longtots1, long_props=longprops1, simple=simpledat)
  return(returnlist)
}


svy_two_variable_interaction_formatter = function(DESIGN, ROW_VAR, COL_VAR, ROW_VAR_TEXT_DESCRIPTION, COL_VAR_TEXT_DESCRIPTION, PRINT_DIGITS){
  print(paste0("########## Currently Running: ", ROW_VAR, " by ", COL_VAR, " ##########"))
  formrow = as.formula(paste0("~", ROW_VAR))
  formcol = as.formula(paste0("~", COL_VAR))
  longtots = cbind.data.frame(melt(svyby(formcol, formrow, design=DESIGN, svytotal, keep.var = FALSE)), round(confint(svyby(formcol, formrow, design=DESIGN, svytotal)), digits=0))
  colnames(longtots) = c("colvar", "rowvar", "n", "ll", "ul")
  longtots$n = round(longtots$n, digits=0)
  longtots$ll = ifelse(longtots$ll<0, 0, longtots$ll)
  widetots = reshape(longtots, idvar = "colvar", timevar = "rowvar", direction = "wide")
  rownames(widetots)=NULL
  longprops = cbind.data.frame(melt(svyby(formcol, formrow, design=DESIGN, svymean, keep.var = FALSE)), confint(svyby(formcol, formrow, design=DESIGN, svymean)))
  colnames(longprops) = c("colvar", "rowvar", "prop", "ll", "ul")
  longprops$prop = round(longprops$prop, digits=PRINT_DIGITS)
  longprops$ll = round(longprops$ll, digits=PRINT_DIGITS)
  longprops$ll = ifelse(longprops$ll<0, 0, longprops$ll)
  longprops$ul = round(longprops$ul, digits=PRINT_DIGITS)
  wideprops = reshape(longprops, idvar = "colvar", timevar = "rowvar", direction = "wide")
  rownames(wideprops)=NULL
  ###############################
  #after wide form datasets created, add in formatted column for long form
  ###############################
  longtots$formattedN = paste0(prettyNum(longtots$n, big.mark=","), " (", prettyNum(longtots$ll, big.mark=","), " - ", prettyNum(longtots$ul, big.mark=","), ")")
  #number formatter #################################
  numformat = function(val) { sub("^(-?)0.", "\\1.", sprintf("%.2f", val)) }
  longprops$formattedP = paste0(numformat(longprops$prop*100), "% (", numformat(longprops$ll*100), "% - ", numformat(longprops$ul*100), "%)")
  # #Rao and Scott adjustment to Pearson's Chi-Sq for survey data
  chitest = svychisq(as.formula(paste0("~", ROW_VAR, "+", COL_VAR)), design=DESIGN, statistic="Chisq", na.rm=TRUE)
  pval = ifelse(chitest$p.value>=.05, paste0("p = ", numformat(chitest$p.value)), paste0("p < .05"))
  chistout = paste0("X^2 = ", numformat(chitest$statistic), ", ", pval)
  longtots$Test = NA
  longtots$Test[1] = chistout
  longprops$Test = NA
  longprops$Test[1] = chistout
  widetots$Test = NA
  widetots$Test[1] = chistout
  wideprops$Test = NA
  wideprops$Test[1] = chistout
  
  longtots1 = cbind.data.frame(Row_Variable = rep(ROW_VAR_TEXT_DESCRIPTION, times=nrow(longtots)), longtots)
  longprops1 = cbind.data.frame(Row_Variable = rep(ROW_VAR_TEXT_DESCRIPTION, times=nrow(longprops)), longprops)
  widetots1 = cbind.data.frame(Row_Variable = rep(ROW_VAR_TEXT_DESCRIPTION, times=nrow(widetots)), widetots)
  wideprops1 = cbind.data.frame(Row_Variable = rep(ROW_VAR_TEXT_DESCRIPTION, times=nrow(wideprops)), wideprops)
  
  #return datasets
  returnlist = list(wide_totals = widetots1, wide_props = wideprops1, long_totals = longtots1, long_props = longprops1)
  return(returnlist)
}

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
#weighted descriptive analyses
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

#set up design framework for already created cross-tabulation function dependent for complex survey design and analysis
subdes = svydesign(id=~1, weight=~1, data=demdat)
#subdes = subset(fulldes, neonate_subpop==1)
summary(subdes)


#find categorical variables to reporte in descriptives
svybyvarlist = descriptive_vars_to_include = c("Patient_Type_Title", "Gender_Title", "Ethnicity_Title", 
                                               "Race.White", "Race.Black", "Race.Asian", "Race.Pacific_Islander", 
                                               "Race.American_Indian", "Race.Other", 
                                               "admit_type", "insurance", "died_in_hospital",
                                               grep("DX_", names(demdat), value=TRUE),
                                               grep("PX_", names(demdat), value=TRUE))

#variable labels
svybyvardescription = c("Patient Type", "Gender", "Ethnicity", 
                        "Race is White", "Race is Black", "Race is Asian", "Race is Pacific Islander", 
                        "Race is American_Indian", "Race is Other", 
                        "Admission Type", "Insurance", "Mortality in Hospital",
                        grep("DX_", names(demdat), value=TRUE),
                        grep("PX_", names(demdat), value=TRUE)
#                        sub("_", " ", grep("DX_", names(demdat), value=TRUE)),
#                        sub("_", " ", grep("PX_", names(demdat), value=TRUE))
                        )

length(svybyvarlist); length(svybyvardescription)

############################################################
############################################################
############################################################
#estimate table descriptives
############################################################
############################################################
############################################################


######################################
#outcome estimates (univariate)
######################################

outcome_oneway_tabs = svy_single_variable_formatter(DESIGN=subdes, ROW_VAR="LOS_category", ROW_VAR_TEXT_DESCRIPTION="Length of Stay", PRINT_DIGITS = 4)
outcome_oneway_tabs

simple_tabs_object = list()
single_tabs_dat_list = list()
for(i in 1:length(svybyvarlist)){
  simple_tabs_object[[i]] = suppressWarnings(svy_single_variable_formatter(DESIGN=subdes, 
                                                                           ROW_VAR=svybyvarlist[i], 
                                                                           ROW_VAR_TEXT_DESCRIPTION=svybyvardescription[i], 
                                                                           PRINT_DIGITS = 4)
                                             )
  single_tabs_dat_list[[i]] = simple_tabs_object[[i]]$simple
}


onewaytabs = rbind.data.frame(outcome_oneway_tabs$simple, do.call("rbind.data.frame", single_tabs_dat_list))
onewaytabs$origdim = dimnames(onewaytabs)[[1]]
onewaytabs[1:40,]

dxnames = read.csv("DX_features_with_at_least_20_neonates_with_EPLOS.csv", header=TRUE)[,-1]
pxnames = read.csv("PX_features_with_at_least_20_neonates_with_EPLOS.csv", header=TRUE)[,-1]
head(dxnames)
head(pxnames)

head(onewaytabs)

onewaytabdesc = merge(
  merge(onewaytabs, subset(dxnames, select=c(icdcode, icd)), by.x=c("varname"), by.y=c("icdcode"), all.x=TRUE), 
  subset(pxnames, select=c(pxicdcode, pxicd)), by.x=c("varname"), by.y=c("pxicdcode"), all.x=TRUE)

onewaytabdesc$label = coalesce(as.character(onewaytabdesc$icd), as.character(onewaytabdesc$pxicd), as.character(onewaytabdesc$Row_Variable))
dput(names(onewaytabdesc))

onewaytabout = subset(onewaytabdesc, select=c(varname, Row_Variable, label, level, n, prop, formatted, formatted_n, formatted_prop))
onewaytabout$level = as.character(onewaytabout$level)
onewaytabout$level[onewaytabout$level=="0"] = "No"
onewaytabout$level[onewaytabout$level=="1"] = "Yes"

onewaytabout = onewaytabout %>%
  arrange(Row_Variable, label, level)

head(onewaytabout)

write.csv(onewaytabout, "one_way_tabulations_all_EPLOS_variables.csv")

######################################
#bivariate data comparisons
######################################



elostabs = list()
elostotswide = list()
elostotslong = list()
elospropswide = list()
elospropslong = list()
for(i in 1:length(svybyvarlist)){
  elostabs[[i]] = suppressWarnings(svy_two_variable_interaction_formatter(DESIGN=subdes, ROW_VAR=svybyvarlist[i], COL_VAR="LOS_category", ROW_VAR_TEXT_DESCRIPTION=svybyvardescription[i], COL_VAR_TEXT_DESCRIPTION="Length of Stay", PRINT_DIGITS=4))
  elostotswide[[i]] = elostabs[[i]]$wide_totals
  elostotslong[[i]] = elostabs[[i]]$long_totals
  elospropswide[[i]] = elostabs[[i]]$wide_props
  elospropslong[[i]] = elostabs[[i]]$long_props
}

tabtotswide = do.call("rbind.data.frame", elostotswide)
tabtotslong = do.call("rbind.data.frame", elostotslong)
tabpropswide = do.call("rbind.data.frame", elospropswide)
tabpropslong = do.call("rbind.data.frame", elospropslong)

glimpse(tabtotslong)
glimpse(tabtotswide)
glimpse(tabpropslong)
glimpse(tabpropswide)


tt=tabtotswide
tp=tabpropswide

grep("n[.]", colnames(tt))
grep("prop[.]", colnames(tp))

rowSums(tt[,grep("n[.]", colnames(tt))])

#varesttabs$rowlevel = gsub(".*\\. (.*)\\..*", "\\1", varesttabs$iavar)
#varesttabs$collevel = sub('.*\\.', '', varesttabs$iavar)

#write.csv(tabtotswide, "EPLOS_selected_features_totals_wide.csv")
#write.csv(tabtotslong, "EPLOS_selected_features_totals_long.csv")
#write.csv(tabpropswide, "EPLOS_selected_features_props_wide.csv")
#write.csv(tabpropslong, "EPLOS_selected_features_props_long.csv")

###################################################################
###################################################################
###################################################################
#SAVE IMAGE
###################################################################
###################################################################
###################################################################
#save.image("PHIS_EPLOS_cohort_description.RData")

dxoneways = subset(onewaytabout, varname %in% grep("DX_", onewaytabout$varname, value=TRUE) & level=="Yes")
pxoneways = subset(onewaytabout, varname %in% grep("PX_", onewaytabout$varname, value=TRUE) & level=="Yes")

head(dxoneways)
dxtabplot = ggplot(dxoneways) +
  geom_bar(aes(x=reorder(label, n), y=n), fill="firebrick", stat="identity") +
  #facet_wrap(~level, nrow=1) +
  coord_flip() + 
  ylab("Frequency") +
  xlab("") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=6),
        panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.minor = element_blank(),
        legend.position = "right")
dxtabplot

head(pxoneways)
pxtabplot = ggplot(pxoneways) +
  geom_bar(aes(x=reorder(label, n), y=n), fill="dodgerblue", stat="identity") +
  #facet_wrap(~level, nrow=1) +
  coord_flip() + 
  ylab("Frequency") +
  xlab("") +
  theme_bw() +
  theme(plot.title = element_text(size=16, lineheight = .8, face="bold", hjust=0),
        strip.text.x = element_text(size=12, face="bold"),
        strip.text.y = element_text(size=12, face="bold"),
        plot.background = element_blank(),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=6),
        panel.grid.major=element_line(colour="grey90", size=.2),
        panel.grid.minor = element_blank(),
        legend.position = "right")
pxtabplot


######################################################################
#write out oneway tabulations DX and PXs for reference in supplementary eTables
######################################################################

alldxoneways = subset(onewaytabout, varname %in% grep("DX_", onewaytabout$varname, value=TRUE))
allpxoneways = subset(onewaytabout, varname %in% grep("PX_", onewaytabout$varname, value=TRUE))

write.csv(alldxoneways, "eTable_EPLOS_DX_tabulations.csv")
write.csv(allpxoneways, "eTable_EPLOS_PX_tabulations.csv")



