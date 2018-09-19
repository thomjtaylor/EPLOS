

rm(list=ls())
dev.off()
###################################################
#script
###################################################
#.R

setwd("/RPROJECTS/Totapally_PICU_KID_EPLOS/PHIS")

#load("/RPROJECTS/Totapally_PICU_KID_EPLOS/PHIS/phis_eplos_cohort.RData")

###################################################
#load packages
###################################################

library(plyr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(keras)
library(scales)
library(survey)
library(DescTools)
library(lubridate)
library(tidyr)
library(caret)

##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
#KEY FUNCTIONS
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################

numformat = function(val) { sub("^(-?)0.", "\\1.", sprintf("%.2f", val)) }

#########################################################
#min max normalization function
#########################################################

min_max_normalization = function(x) {
  normalized = (x-min(x, na.rm=TRUE))/(max(x, na.rm=TRUE)-min(x, na.rm=TRUE))
  return(normalized)
}


##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
#LOAD FULL DATA FILES NEEDED
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################

list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/")

list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_readmit_all_2017/")

list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jan_2017/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_feb_mar_2017/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_apr_may_2017/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jun_jul_2017/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_aug_2017/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_sep_2017/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_oct_2017/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_nov_2017/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_dec_2017/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jan_feb_2016/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_mar_apr_2016/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_may_jun_2016/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jul_aug_2016/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_sep_oct_2016/")
list.files("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_nov_dec_2016/")


#all 2017 readmissions
#readmit2017 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_readmit_all_2017/ReadmitPatientData.csv", header=TRUE)

#patient abstract with patient data for 2016-2017 (note that there may be some duplication? of records between months from cohort builder--these are removed)
paall = unique(rbind.data.frame(pa1 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jan_2017/PatientAbstract.csv", header=TRUE),
                                pa23 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_feb_mar_2017/PatientAbstract.csv", header=TRUE),
                                pa45 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_apr_may_2017/PatientAbstract.csv", header=TRUE),
                                pa67 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jun_jul_2017/PatientAbstract.csv", header=TRUE),
                                pa8 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_aug_2017/PatientAbstract.csv", header=TRUE),
                                pa9 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_sep_2017/PatientAbstract.csv", header=TRUE),
                                pa10 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_oct_2017/PatientAbstract.csv", header=TRUE),
                                pa11 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_nov_2017/PatientAbstract.csv", header=TRUE),
                                pa12 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_dec_2017/PatientAbstract.csv", header=TRUE)
                                ,
                                pa_2016_1_2 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jan_feb_2016/PatientAbstract.csv", header=TRUE),
                                pa_2016_3_4 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_mar_apr_2016/PatientAbstract.csv", header=TRUE),
                                pa_2016_5_6 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_may_jun_2016/PatientAbstract.csv", header=TRUE),
                                pa_2016_7_8 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jul_aug_2016/PatientAbstract.csv", header=TRUE),
                                pa_2016_9_10 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_sep_oct_2016/PatientAbstract.csv", header=TRUE),
                                pa_2016_11_12 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_nov_dec_2016/PatientAbstract.csv", header=TRUE)
))


#patient dx 2016-2017 (note that there may be some duplication? of records between months from cohort builder--these are removed)
dxall = unique(rbind.data.frame(dx1 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jan_2017/Diagnosis.csv", header=TRUE),
                                dx23 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_feb_mar_2017/Diagnosis.csv", header=TRUE),
                                dx45 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_apr_may_2017/Diagnosis.csv", header=TRUE),
                                dx67 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jun_jul_2017/Diagnosis.csv", header=TRUE),
                                dx8 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_aug_2017/Diagnosis.csv", header=TRUE),
                                dx9 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_sep_2017/Diagnosis.csv", header=TRUE),
                                dx10 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_oct_2017/Diagnosis.csv", header=TRUE),
                                dx11 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_nov_2017/Diagnosis.csv", header=TRUE),
                                dx12 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_dec_2017/Diagnosis.csv", header=TRUE)
                                ,
                                dx_2016_1_2 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jan_feb_2016/Diagnosis.csv", header=TRUE),
                                dx_2016_3_4 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_mar_apr_2016/Diagnosis.csv", header=TRUE),
                                dx_2016_5_6 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_may_jun_2016/Diagnosis.csv", header=TRUE),
                                dx_2016_7_8 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jul_aug_2016/Diagnosis.csv", header=TRUE),
                                dx_2016_9_10 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_sep_oct_2016/Diagnosis.csv", header=TRUE),
                                dx_2016_11_12 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_nov_dec_2016/Diagnosis.csv", header=TRUE)
))

suic_events = data.frame(table(dxall$Dx_Title_.ICD.[grep("suic", tolower(dxall$Dx_Title_.ICD.))])) %>% arrange(-Freq)


#patient px 2016-2017 (note that there may be some duplication? of records between months from cohort builder--these are removed)
pxall = unique(rbind.data.frame(px1 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jan_2017/Procedure.csv", header=TRUE),
                                px23 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_feb_mar_2017/Procedure.csv", header=TRUE),
                                px45 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_apr_may_2017/Procedure.csv", header=TRUE),
                                px67 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jun_jul_2017/Procedure.csv", header=TRUE),
                                px8 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_aug_2017/Procedure.csv", header=TRUE),
                                px9 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_sep_2017/Procedure.csv", header=TRUE),
                                px10 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_oct_2017/Procedure.csv", header=TRUE),
                                px11 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_nov_2017/Procedure.csv", header=TRUE),
                                px12 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_dec_2017/Procedure.csv", header=TRUE),
                                px_2016_1_2 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jan_feb_2016/Procedure.csv", header=TRUE),
                                px_2016_3_4 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_mar_apr_2016/Procedure.csv", header=TRUE),
                                px_2016_5_6 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_may_jun_2016/Procedure.csv", header=TRUE),
                                px_2016_7_8 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_jul_aug_2016/Procedure.csv", header=TRUE),
                                px_2016_9_10 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_sep_oct_2016/Procedure.csv", header=TRUE),
                                px_2016_11_12 = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/PHIS/cohort_builder/inpatient_obs_cases_nov_dec_2016/Procedure.csv", header=TRUE)))

#load National Quality Forum NQF Boston Childrens readmission materials for Planned Procedures
list.files("/DATA/Salyakina_NCHS_ReadmRisk/National_Quality_Forum_NQF_Boston_childrens_readmission_materials")
nqf_ccis = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/National_Quality_Forum_NQF_Boston_childrens_readmission_materials/CCIs_Boston_Childrens_All_Cause_Readmission_ICD9_and_ICD10_categorizations.csv", header=TRUE)
nqf_chemo = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/National_Quality_Forum_NQF_Boston_childrens_readmission_materials/Chemotherapy_Boston_Childrens_All_Cause_Readmission_ICD9_and_ICD10_DX_and_HCPCS_codes.csv", header=TRUE)
nqf_menthealth = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/National_Quality_Forum_NQF_Boston_childrens_readmission_materials/Mental_Health_Boston_Childrens_All_Cause_Readmission_ICD9_and_ICD10_codes.csv", header=TRUE)
nqf_planned_pxs = read.csv("/DATA/Salyakina_NCHS_ReadmRisk/National_Quality_Forum_NQF_Boston_childrens_readmission_materials/Planned_Procedures_Boston_Childrens_All_Cause_Readmission_ICD9_and_ICD10_categorizations.csv", header=TRUE)

####################################################
#FORMATTING and FEATURE ENGINEERING
####################################################

#discharge status
discharge_dispositions_in_cohort = data.frame(table(paall$Disposition_Title)) %>% arrange(-Freq)
discharge_dispositions_in_cohort

paall$disch_type = NA
paall$disch_type[paall$Disposition_Title=="Discharge to Home or Self Care (Routine Discharge)"] = "a. Home"
paall$disch_type[paall$Disposition_Title=="Discharged/Transferred to Home under Care of Organized Home Health Service Organization in Anticipation of Covered Skilled Care"] = "b. Home Health Care"
paall$disch_type[paall$Disposition_Title=="Discharged/Transferred to a Psychiatric Hospital or Psychiatric Distinct Part Unit of a Hospital"] = "c. Psychiatric Care"	
paall$disch_type[paall$Disposition_Title=="Expired"] = "d. In-Hospital Mortality"	
paall$disch_type[is.na(paall$disch_type)==TRUE] = "e. Other"
paall$disch_type = factor(paall$disch_type)
table(paall$disch_type, useNA = "ifany")

#priority of admission
table(paall$Priority_Of_Admission_Title)
paall$admit_type = NA
paall$admit_type[paall$Priority_Of_Admission_Title=="Elective"] = "a. Elective"
paall$admit_type[paall$Priority_Of_Admission_Title=="Emergency"|paall$Priority_Of_Admission_Title=="Urgent"|paall$Priority_Of_Admission_Title=="Trauma"] = "b. ED/UC"
paall$admit_type[paall$Priority_Of_Admission_Title=="Newborn"] = "c. Newborn"
paall$admit_type[paall$Priority_Of_Admission_Title=="Undefined"|paall$Priority_Of_Admission_Title=="Information Not Available"] = "d. No Information"
paall$admit_type[is.na(paall$admit_type)==TRUE] = "d. No Information"
paall$admit_type = factor(paall$admit_type)
table(paall$admit_type, useNA = "ifany")


#primary source of payment title recoding
data.frame(table(paall$Primary_Source_Of_Payment_Title)) %>% arrange(-Freq)
dput(levels(paall$Primary_Source_Of_Payment_Title))
paall$insurance = factor(as.character(mapvalues(paall$Primary_Source_Of_Payment_Title,
                                                from=c("Charity", "CHIP", "Commercial HMO", "Commercial Other", "Commercial PPO", 
                                                       "Hospital chose not to bill for this encounter", "In-State Medicaid (managed care)", 
                                                       "In-State Medicaid (other)", "Medicare", "Other Government", 
                                                       "Other Payor", "Out-of-State Medicaid (all)", "Self Pay", "TRICARE", 
                                                       "Unknown"),
                                                to=c("d. Other", "a. Government", "b. Commercial", "b. Commercial", "b. Commercial", 
                                                     "d. Other", "a. Government", 
                                                     "a. Government", "a. Government", "a. Government", 
                                                     "d. Other", "a. Government", "c. Self Pay", "a. Government", 
                                                     "e. Unknown"))))
with(paall, table(Primary_Source_Of_Payment_Title, insurance))
table(paall$insurance)

####################################################
#REDUCE full datasets to only variables on interest
####################################################

dput(names(paall))

pa = unique(subset(paall, CTC_Flag!="N", select=c("Hospital_Number", "Hospital_City", "Hospital_Name", "Campus_Name", 
                                                  "Medical_Record_Number", "Discharge_ID", 
                                                  #"Billing_Number", 
                                                  "Patient_Type_Title", "Admit_Age_In_Days", "Admit_Age_In_Years", 
                                                  "Gender_Title", "Ethnicity_Title", 
                                                  "Race.White", "Race.Black", "Race.Asian", "Race.Pacific_Islander", "Race.American_Indian", "Race.Other",
                                                  "Length_Of_Stay", 
                                                  "disch_type", "admit_type",
                                                  "Complex_Chronic_Condition_Flag", 
                                                  #excluded non-necessary flags for now
                                                  #"ED_Charge_Flag", "Premature_And_Neonatal_Flag",
                                                  #"NICU_Flag", "ICU_Flag", "Mechanical_Vent_Flag", "ECMO_Flag", "TPN_Flag", "Operating_Room_Charge_Flag", "Infection_Flag", "Medical_Complication_Flag", "Surgical_Complication_Flag", 
                                                  #"Cardiovascular_Flag", "Gastrointestinal_Flag", "Hematologic_or_Immunologic_Flag", "Malignancy_Flag", "Metabolic_Flag", 
                                                  #"Neurologic_and_Neuromuscular_Flag", "Congenital_or_Genetic_Defect_Flg", "Renal_and_Urologic_Flag", "Respiratory_Flag",
                                                  "Admit_Date", "Discharge_Date", 
                                                  "Principal_Dx_.ICD.", "Principal_Dx_Title_.ICD.",  
                                                  #appears to be fairly dirty data on dxs that are present on admit--may be due to how documented in source systems and inconsistences that arise
                                                  #"Principal_Dx_Present_On_Admit", "Principal_Dx_POA_Admit_Title", 
                                                  "Principal_Px_.ICD.", "Principal_Px_Title_.ICD.", 
                                                  #"Admit_Dx_.ICD.", "Admit_Dx_Title_.ICD.", 
                                                  "insurance", "Census_Region", "Zip_Code", "Urban_Flag", "Median_Household_Income", "Predicted_Median_Household_Income"
)
)
)


########################################################################
#create key exclusion criteria
########################################################################

pa$age_yrs = pa$Admit_Age_In_Days/365
pa = pa[pa$age_yrs>=0 & pa$age_yrs<18 & pa$Gender_Title!="UNKNOWN" & pa$Gender_Title!="Unknown",]
nrow(pa)

########################################################################
#clean up Patient Abstract cohort (e.g., encode missingness, etc.)
########################################################################


#decode missing 
pa[pa==""] = NA
pa[pa=="-1"] = NA
pa$Principal_Dx_.ICD.[pa$Principal_Dx_.ICD.=="-------"] = NA
pa$Principal_Dx_.ICD.[pa$Principal_Dx_.ICD.=="@@@@@@@"] = NA

pa$Principal_Px_.ICD.[pa$Principal_Px_.ICD.=="-1"] = NA

pa = pa %>% mutate_if(is.factor, as.character) %>% mutate_if(is.character, as.factor)

summary(pa)
str(pa)

####################################################
####################################################
####################################################
#FORMATTING
####################################################
####################################################
####################################################

#lubridate dates
pa$Admit_Date = mdy(as.character(pa$Admit_Date))
pa$Discharge_Date = mdy(as.character(pa$Discharge_Date))

#exclude patients that may be errors or electives anyway

pa$LOS = as.numeric(pa$Discharge_Date - pa$Admit_Date)

#############################################################
#############################################################
#############################################################
#integrate indicators of probably planned procedures and diagnoses
#############################################################
#############################################################
#############################################################

grep("nqf", ls(), value=TRUE)
summary(nqf_ccis)
summary(nqf_menthealth)
summary(nqf_chemo)
summary(nqf_planned_pxs)

nqf_chemo$padded_chemo_dx_or_hcpcs = stringr::str_pad(nqf_chemo$chemotherapy_dx_or_hcpcs, 7, pad = "0")
nqf_chemo$nqf_planned_chemo_hcpcs = 1

head(nqf_planned_pxs)
max(nchar(as.character(nqf_planned_pxs$hcpcs)))
table(nqf_planned_pxs$hcpcs)
nqf_planned_pxs$nqf_planned_hcpcs = 1

#left outer join pa to nqf planned chemo codes
pa = merge(pa, subset(nqf_chemo, select=c(padded_chemo_dx_or_hcpcs, nqf_planned_chemo_hcpcs)), by.x=c("Principal_Px_.ICD."), by.y=c("padded_chemo_dx_or_hcpcs"), all.x=TRUE) 
pa$nqf_planned_chemo_hcpcs[is.na(pa$nqf_planned_chemo_hcpcs)==TRUE] = 0

#left outer join to nqf planned chemo codes
pa = merge(pa, subset(nqf_planned_pxs, select=c(hcpcs, nqf_planned_hcpcs)), by.x=c("Principal_Px_.ICD."), by.y=c("hcpcs"), all.x=TRUE) 
pa$nqf_planned_hcpcs[is.na(pa$nqf_planned_hcpcs)==TRUE] = 0

data.frame(table(pa$nqf_planned_hcpcs, pa$nqf_planned_chemo_hcpcs))


#create composite indicator that highlights a Potentially Planned Event (PPE)
pa$PPE = ifelse(pa$admit_type=="a. Elective"|pa$nqf_planned_chemo_hcpcs==1|pa$nqf_planned_hcpcs==1, 1, 0)

###############################################################################
###############################################################################
###############################################################################
#encoding death and 
#unplanned readmissions in cohort at specified time
###############################################################################
###############################################################################
###############################################################################

#########################################################
#Died
#########################################################

pa$died_in_hospital = ifelse(pa$disch_type=="d. In-Hospital Mortality", 1, 0)
table(pa$died_in_hospital)

#########################################################
#Length of Stay Indicators
#########################################################


pa$LOSmonths = cut(pa$LOS, breaks = c(0, 30, 60, 90, 120, 150, 180, 1000), right=TRUE, include.lowest = TRUE) 
table(pa$LOSmonths)

pa$eplos = ifelse(pa$LOS>=180, 1, 0)
table(pa$eplos)

#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#capture only diagnoses of interest (>50 patients who had procedure in EPLOS group & average time to procedure was less than 30 days)
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################

head(dxall)
#decode missing 
dxall[dxall==""] = NA
dxall[dxall=="-1"] = NA
dxall$Principal_Dx_.ICD.[dxall$Dx_Code_.ICD.=="-------"] = NA
dxall$Principal_Dx_.ICD.[dxall$Dx_Code_.ICD.=="@@@@@@@"] = NA

#create new necessary codes for reshaping and cleanup of dxs
dxall$encntr_id = factor(paste0(dxall$Medical_Record_Number, "--", dxall$Discharge_ID, "--", dxall$Billing_Number))
dxall$rowid = as.numeric(dxall$encntr_id)
dxall$icdcode = paste0("DX_", dxall$Dx_Code_.ICD.)
dxall$icd = paste0(dxall$Dx_Code_.ICD., " -- ", dxall$Dx_Title_.ICD.)
dxall$icd10 = 1

#lubridate dates
dxall$Discharge_Date = mdy(as.character(dxall$Discharge_Date))

#but, since no date of diagnosis, using present on admit instead!!!!!!!!!!!!!!!!!
#use only DXs present on admit since can't be sure when assigned afterward
table(dxall$Dx_Present_On_Admit_Title, dxall$Dx_Present_On_Admit)

poadx = unique(merge(dxall[dxall$Dx_Present_On_Admit_Title=="Yes",], subset(pa, select=c(Discharge_ID, LOSmonths, eplos)), by=c("Discharge_ID")))


#############################################################
#############################################################
#############################################################
#assess ICD potential death indicators
#############################################################
#############################################################
#############################################################

possible_death_dxs = data.frame(table(poadx$icd[grep("died|death|mort|expire", tolower(poadx$icd))])) %>% arrange(-Freq)
possible_death_dxs

#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#Diagnosis feature engineering (selection)
#find the most frequent diagnoses for each outcome and combine
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################

eplosdxs = subset(poadx, eplos==1) %>%
  group_by(icdcode, icd) %>%
  summarize(total_pats_with_dx = n_distinct(Medical_Record_Number), n = n()) %>% 
  filter(icd!="Z66 -- Do not resuscitate" & icd!="Z515 -- Encounter for palliative care" & icd!="G9382 -- Brain death") %>% 
  arrange(-total_pats_with_dx) %>% 
  mutate(Event = "DX") %>%
  group_by(Event) %>%
  mutate(rank = row_number()) %>%
  filter(total_pats_with_dx>=50) 
head(eplosdxs)
nrow(eplosdxs)
data.frame(eplosdxs) %>% top_n(1000, wt=total_pats_with_dx)

write.csv(eplosdxs, "DX_features_with_at_least_50_EPLOS_patients.csv")

#############################################################
#Create an internal list of unique diagnoses to include in the feature space
#############################################################

nrow(eplosdxs)
head(eplosdxs)
unique_eplosdxs = unique(subset(eplosdxs, select=c(icdcode, icd)))
nrow(unique_eplosdxs)
head(unique_eplosdxs)

#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#capture only procedures of interest (>50 patients who had procedure in EPLOS group & average time to procedure was less than 30 days)
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################

head(pxall)
#decode missing 
pxall[pxall==""] = NA
pxall[pxall=="-1"] = NA
pxall$Principal_Px_.ICD.[pxall$Px_Code_.ICD.=="-------"] = NA
pxall$Principal_Px_.ICD.[pxall$Px_Code_.ICD.=="@@@@@@@"] = NA

#create new necessary codes for reshaping and cleanup of dxs
pxall$encntr_id = factor(paste0(pxall$Medical_Record_Number, "--", pxall$Discharge_ID, "--", pxall$Billing_Number))
pxall$rowid = as.numeric(pxall$encntr_id)
pxall$pxicdcode = paste0("PX_", pxall$Px_Code_.ICD.)
pxall$pxicd = paste0(pxall$Px_Code_.ICD., " -- ", pxall$Px_Title_.ICD.)
pxall$pxicd10 = 1

head(pxall)

#lubridate dates
pxall$Date_Of_Service = mdy(as.character(pxall$Date_Of_Service))
pxall$Discharge_Date = mdy(as.character(pxall$Discharge_Date))


pxmer = unique(merge(pxall, subset(pa, select=c(Discharge_ID, LOSmonths, eplos, Admit_Date)), by=c("Discharge_ID")))
pxmer$days_to_px_from_admit = as.numeric(pxmer$Date_Of_Service - pxmer$Admit_Date)
table(pxmer$days_to_px_from_admit)

#############################################################
#NOTE!!!! PXs from discharge IDs that were schedule long before actual admit date are likely electives or errors, and should be removed from cohort
#############################################################

elective_discharge_ids_from_pxs = unique(subset(pxmer, days_to_px_from_admit<0, select=c(Discharge_ID)))
elective_discharge_ids_from_pxs$likely_elective = 1

pxmersub = subset(pxmer, days_to_px_from_admit>=0)
nrow(pxmersub)

#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#Procedures feature engineering (selection)
#find the most frequent procedures for each outcome and the average times within 30 days of admission for patient
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################

eplospxs = subset(pxmersub, eplos==1) %>%
  group_by(pxicdcode, pxicd) %>%
  summarize(total_pats_with_px = n_distinct(Medical_Record_Number), 
            n_procedures = n(), 
            min_days_to_px = min(days_to_px_from_admit, na.rm=TRUE),
            median_days_to_px = median(days_to_px_from_admit, na.rm=TRUE),
            max_days_to_px = max(days_to_px_from_admit, na.rm=TRUE),
            mean_days_to_px = mean(days_to_px_from_admit, na.rm=TRUE),
            sd_days_to_px = sd(days_to_px_from_admit, na.rm=TRUE)) %>% 
  arrange(-total_pats_with_px) %>% 
  mutate(Event = "PX") %>%
  group_by(Event) %>%
  mutate(rank = row_number()) %>%
  filter(total_pats_with_px>=50 & median_days_to_px<=30) 
head(eplospxs)
nrow(eplospxs)
#View(eplospxs)
data.frame(eplospxs) %>% top_n(100, wt=total_pats_with_px)
glimpse(eplospxs)

write.csv(eplospxs, "PX_features_with_at_least_50_EPLOS_patients.csv")

#############################################################
#Create an internal list of unique diagnoses to include in the feature space
#############################################################

nrow(eplospxs)
head(eplospxs)
unique_eplospxs = unique(subset(eplospxs, select=c(pxicdcode, pxicd)))
nrow(unique_eplospxs)
head(unique_eplospxs)

####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
#create reference cohort
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################

#don't use principal dx or px as those may occur long AFTER the 30 day window of interest for predicting 6+ month stays (EPLOS)

dput(names(pa))

refpa = merge(subset(pa, select=c("Medical_Record_Number", "Discharge_ID", "Patient_Type_Title", "Gender_Title", "Ethnicity_Title", 
                                  "Race.White", "Race.Black", "Race.Asian", "Race.Pacific_Islander", 
                                  "Race.American_Indian", "Race.Other", "disch_type", 
                                  "admit_type", "Complex_Chronic_Condition_Flag", "Admit_Date",
                                  "Principal_Dx_.ICD.", "Principal_Dx_Title_.ICD.", "Principal_Px_.ICD.", "Principal_Px_Title_.ICD.", 
                                  "Discharge_Date", 
                                  "insurance",
                                  "age_yrs", "LOS", 
                                  "nqf_planned_chemo_hcpcs", "nqf_planned_hcpcs", 
                                  "PPE", "died_in_hospital", "LOSmonths", "EPLOS", "eplos")),
              elective_discharge_ids_from_pxs,
              by=c("Discharge_ID"), all.x=TRUE) %>%
  filter(is.na(likely_elective)==TRUE & nqf_planned_chemo_hcpcs==0 & nqf_planned_hcpcs==0 & PPE==0)

refpa = refpa[is.na(refpa$Principal_Dx_.ICD.)==FALSE & is.na(refpa$Principal_Dx_.ICD.)==FALSE & is.na(refpa$LOSmonths)==FALSE,]
refpa = subset(refpa, select=-c(likely_elective, 
                                nqf_planned_chemo_hcpcs, 
                                nqf_planned_hcpcs
                                #, 
                                #Principal_Dx_.ICD., 
                                #Principal_Dx_Title_.ICD., 
                                #Principal_Px_.ICD., 
                                #Principal_Px_Title_.ICD.
                                ))
summary(refpa)
glimpse(refpa)

####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
#create discharge level dxs from selected DXs with 50 or more EPLOS patients
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################


dxsub = unique(merge(unique(subset(poadx, select=c(Discharge_ID, icdcode))),
                         subset(unique_eplosdxs, select=c(icdcode)),
                         by=c("icdcode")))
nrow(dxsub)
dxsub$icd10 = 1
head(dxsub)
length(unique(dxsub$icdcode))

#############################################################
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#reshape dxs to wide
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#############################################################

dxwide = reshape2::dcast(dxsub, Discharge_ID ~ icdcode, value.var = "icd10")
nrow(dxwide); ncol(dxwide)

#change NA to 0
dxwide[is.na(dxwide)] = 0
dxwide[1:10,1:30]
nrow(dxwide)

dx_column_unique_values = apply(dxwide, 2, function(x) length(unique(x)))
#all columns have variance
table(dx_column_unique_values)

####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
#create discharge level PXs from selected PXs with 50 or more EPLOS patients who had procedures with median of under 30 days
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################


pxsub = unique(merge(unique(subset(pxmersub, select=c(Discharge_ID, pxicdcode))),
                     subset(unique_eplospxs, select=c(pxicdcode)),
                     by=c("pxicdcode")))
nrow(pxsub)
pxsub$pxicd10 = 1
head(pxsub)
length(unique(pxsub$pxicdcode))

#############################################################
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#reshape pxs to wide
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#############################################################

pxwide = reshape2::dcast(pxsub, Discharge_ID ~ pxicdcode, value.var = "pxicd10")
nrow(pxwide); ncol(pxwide)

#change NA to 0
pxwide[is.na(pxwide)] = 0
pxwide[1:10,]
nrow(pxwide)

px_column_unique_values = apply(pxwide, 2, function(x) length(unique(x)))
#all columns have variance
table(px_column_unique_values)


####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
#CREATE FULL MERGED COHORT with DXs and PXs
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################

fmc = merge(merge(refpa, dxwide, by=c("Discharge_ID"), all.x=TRUE), pxwide, by=c("Discharge_ID"), all.x=TRUE)

nrow(refpa)
nrow(fmc)

#####################################################################
#replace NA where no dx or px in lists selected for EPLOS 
#note: Principal DX and PXs in refpa data frame that lack a value will be imputed with 0 as well!
#####################################################################

fmc = fmc %>% mutate_if(is.factor, as.character)
fmc[is.na(fmc)] = 0
fmc = fmc %>% mutate_if(is.character, as.factor)

glimpse(fmc)

write.csv(fmc, "phis_eplos_2016_2017_cohort.csv")



#######################################################################
#write a temporary file to develop further code on which is smallar and takes less time to load/develop with
#######################################################################
fmceploscases = subset(fmc, eplos==1)
nrow(fmceploscases)
fmceplosNONcases = subset(fmc, eplos==0)[1:10870,]
nrow(fmceplosNONcases)

temp_phis_eplos_cohort = rbind.data.frame(fmceploscases, fmceplosNONcases)
nrow(temp_phis_eplos_cohort)

write.csv(temp_phis_eplos_cohort, "temp_phis_eplos_2016_2017_cohort.csv")



#############################################################
#clean up data for output
#############################################################

#######################################################
#######################################################
#######################################################
#data cleaning and FEATURE SELECTION FOR DEATH
#######################################################
#######################################################
#######################################################


dput(names(refpa))

variables_to_include = c("Discharge_ID", "Gender_Title", "Ethnicity_Title", "Race.White", "Race.Black", 
                         "Race.Asian", "Race.Pacific_Islander", "Race.American_Indian", 
                         "Race.Other", "admit_type", "Complex_Chronic_Condition_Flag", 
                         "Admit_Date", "insurance", "age_yrs", "LOS", "died_in_hospital", "LOSmonths", 
                         "eplos",
                         grep("DX_", names(fmc), value=TRUE),
                         grep("PX_", names(fmc), value=TRUE))

variables_to_include

##########################################################
#SCALE CONTINUOUS VARIABLES BEFORE K-Fold Cross Validation
##########################################################

age_yrs_descriptives = psych::describe(fmc$age_yrs, na.rm=TRUE)
age_yrs_descriptives

fmc$age_yrs = min_max_normalization(fmc$age_yrs)
psych::describe(fmc$age_yrs)

#########################################################
#replace missing DXs with 0 (since patient did not have a DX in PHIS DX files)
#########################################################
cohdiedsub[is.na(cohdiedsub)] = 0

#########################################################
#########################################################
#########################################################
#one-hot encode died dataset
#########################################################
#########################################################
#########################################################

glimpse(cohdiedsub[,1:25])

#load caret first, don't call directly from caret with ::, seems not to work
died_dummyset = dummyVars("~ . ", data=cohdiedsub)
died_ohed = data.frame(predict(died_dummyset, newdata=cohdiedsub))

#########################################################
#summarize died cohort
#########################################################
names(died_ohed)
summary(died_ohed[,1:30])
glimpse(died_ohed[,1:50])
nrow(died_ohed)
ncol(died_ohed)

table(died_ohed$died_in_hospital)
prop.table(table(died_ohed$died_in_hospital))

