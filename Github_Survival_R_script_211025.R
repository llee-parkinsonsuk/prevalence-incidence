# Set working directory
setwd("C:/Users/2877245G/OneDrive - University of Glasgow/Documents/MD/Incidence and Prevalence/Analysis/Survival/CPRD_HES data/Survival Analysis")

# Turn off scientific notation
options(scipen=999)

# Install and load required packages
pacman::p_load(
  dplyr,
  survival,
  survminer,
  ggplot2,
  tibble,
  lubridate,
  ggsurvfit,
  gtsummary,
  tidycmprsk,
  tidyverse,
  rstatix)

### KAPLAN MEIER SURVIVAL CURVES FOR THE 8 PARKINSONIAN DISORDERS

### PD

# Read in PD dataset
PD <- readRDS("PD_survival_170125.RDS")

# Create a survival object for use as the response in the model formula.
s1_PD <- Surv(PD$time, PD$outcome)

# Use the survfit() function and stratify by group (i.e. disease)
s1_PD <- survfit(s1_PD ~ PD$group)

# Plot unadjusted Kaplan-Meier survival curves for PD cases and controls
plt_PD <- survfit2(Surv(time, outcome) ~ group, data = PD) %>% 
  ggsurvfit(linewidth = 1) +
  labs(
    y = "Percentage Survival",
    title = "Parkinson's disease",
    x = NULL
  ) + 
  add_confidence_interval() +
  scale_ggsurvfit() + 
  add_quantile(y_value = 0.5, color = "gray50", linewidth = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0, 21, by = 3),
                     minor_breaks = seq(0, 21, 1)) +
  scale_color_manual(values = c('dark blue', 'red'),
                     labels = c('Control', 'Case')) +
  scale_fill_manual(values = c('dark blue', 'red'),
                    labels = c('Control', 'Case')) +
  theme(plot.title = element_text(face = "bold", family="serif", size = 14, hjust = 0.5)) +
  theme(legend.position = c(0.9, 0.9)) +
  theme(axis.title=element_text(size=10,  family="serif"), axis.text = element_text(size=10,  family="serif"), legend.text = element_text(size=10,  family="serif"))

### MSA

# Read in subsetted MSA data
MSA <- readRDS("MSA_new_survival_120325.RDS")

# create a survival object for use as the response in the model formula. 
s1_MSA <- Surv(MSA$time, MSA$outcome)

# Use the survfit() function and stratify by group (i.e. MSA vs control)
s1_MSA <- survfit(s1_MSA ~ MSA$group)

# Plot Kaplan-Meier survival curves for MSA cases and controls
plt_MSA <- survfit2(Surv(time, outcome) ~ group, data = MSA) %>% 
  ggsurvfit(linewidth = 1) +
  labs(
    y = NULL,
    title = "Multiple System Atrophy",
    x = NULL
  ) + 
  add_confidence_interval() +
  scale_ggsurvfit() + 
  add_quantile(y_value = 0.5, color = "gray50", linewidth = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0, 21, by = 3),
                     minor_breaks = seq(0, 21, 1)) +
  scale_color_manual(values = c('dark blue', 'red'),
                     labels = c('Control', 'Case')) +
  scale_fill_manual(values = c('dark blue', 'red'),
                    labels = c('Control', 'Case')) +
  theme(plot.title = element_text(face = "bold", family="serif", size = 14, hjust = 0.5)) +
  theme(legend.position = c(0.9, 0.9)) +
  theme(axis.title=element_text(size=10,  family="serif"), axis.text = element_text(size=10,  family="serif"), legend.text = element_text(size=10,  family="serif"))

### 

# Read in subsetted PSP data
PSP <- readRDS("PSP_new_survival_250225.RDS")

# create a survival object for use as the response in the model formula. 
s1_PSP <- Surv(PSP$time, PSP$outcome)

# Use the survfit() function and stratify by group (i.e. PSP vs control)
s1_PSP <- survfit(s1_PSP ~ PSP$group)

# Plot Kaplan-Meier survival curves for PSP cases and controls
plt_PSP <- survfit2(Surv(time, outcome) ~ group, data = PSP) %>% 
  ggsurvfit(linewidth = 1) +
  labs(
    y = "Percentage Survival",
    title = "Progressive Supranuclear Palsy",
    x = NULL
  ) + 
  add_confidence_interval() +
  scale_ggsurvfit() + 
  add_quantile(y_value = 0.5, color = "gray50", linewidth = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0, 21, by = 3),
                     minor_breaks = seq(0, 21, 1)) +
  scale_color_manual(values = c('dark blue', 'red'),
                     labels = c('Control', 'Case')) +
  scale_fill_manual(values = c('dark blue', 'red'),
                    labels = c('Control', 'Case')) +
  theme(plot.title = element_text(face = "bold", family="serif", size = 14, hjust = 0.5)) +
  theme(legend.position = c(0.9, 0.9)) +
  theme(axis.title=element_text(size=10,  family="serif"), axis.text = element_text(size=10,  family="serif"), legend.text = element_text(size=10,  family="serif"))

### CBS

# Read in subsetted CBS data
CBS <- readRDS("CBS_survival_170125.RDS")

# create a survival object for use as the response in the model formula. 
s1_CBS <- Surv(CBS$time, CBS$outcome)

# Use the survfit() function and stratify by group (i.e. CBS vs control)
s1_CBS <- survfit(s1_CBS ~ CBS$group)

# Plot Kaplan-Meier survival curves for CBS cases and controls
plt_CBS <- survfit2(Surv(time, outcome) ~ group, data = CBS) %>% 
  ggsurvfit(linewidth = 1) +
  labs(
    y = NULL,
    title = "Corticobasal Syndrome",
    x = NULL
  ) + 
  add_confidence_interval() +
  scale_ggsurvfit() + 
  scale_x_continuous(breaks = seq(0, 21, by = 3),
                     minor_breaks = seq(0, 21, 1)) +
  add_quantile(y_value = 0.5, color = "gray50", linewidth = 1) +
  theme_minimal() +
  scale_color_manual(values = c('dark blue', 'red'),
                     labels = c('Control', 'Case')) +
  scale_fill_manual(values = c('dark blue', 'red'),
                    labels = c('Control', 'Case')) +
  theme(plot.title = element_text(face = "bold", family="serif", size = 14, hjust = 0.5)) +
  theme(legend.position = c(0.9, 0.9)) +
  theme(axis.title=element_text(size=10,  family="serif"), axis.text = element_text(size=10,  family="serif"), legend.text = element_text(size=10,  family="serif"))

### DLB

# Read in subsetted DLB data
DLB <- readRDS("DLB_new_survival_CPRDonly_240225.RDS")

# create a survival object for use as the response in the model formula. 
s1_DLB <- Surv(DLB$time, DLB$outcome)

# Use the survfit() function and stratify by group (i.e. DLB vs control)
s1_DLB <- survfit(s1_DLB ~ DLB$group)

# Plot Kaplan-Meier survival curves for DLB cases and controls
plt_DLB <- survfit2(Surv(time, outcome) ~ group, data = DLB) %>% 
  ggsurvfit(linewidth = 1) +
  labs(
    y = "Percentage Survival",
    title = "Dementia with Lewy Bodies"
  ) + 
  add_confidence_interval() +
  scale_ggsurvfit() + 
  add_quantile(y_value = 0.5, color = "gray50", linewidth = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0, 21, by = 3),
                     minor_breaks = seq(0, 21, 1)) +
  scale_color_manual(values = c('dark blue', 'red'),
                     labels = c('Control', 'Case')) +
  scale_fill_manual(values = c('dark blue', 'red'),
                    labels = c('Control', 'Case')) +
  theme(plot.title = element_text(face = "bold", family="serif", size = 14, hjust = 0.5)) +
  theme(legend.position = c(0.9, 0.9)) +
  theme(axis.title=element_text(size=10,  family="serif"), axis.text = element_text(size=10,  family="serif"), legend.text = element_text(size=10,  family="serif")) +
  theme(axis.title.x = element_blank())

### VP 

VP <- readRDS("VP_new_survival_130325.RDS")

# Use the Surv() function from the survival package to create a survival object 
s1_VP <- Surv(VP$time, VP$outcome)

# Use the survfit() function and stratify by group
s1_VP <- survfit(s1_VP ~ VP$group)

# Plot Kaplan-Meier survival curves for VP cases and controls
plt_VP <- survfit2(Surv(time, outcome) ~ group, data = VP) %>% 
  ggsurvfit(linewidth = 1) +
  labs(
    y = NULL,
    title = "Vascular parkinsonism"
  ) + 
  add_confidence_interval() +
  scale_ggsurvfit(x_scales=list(breaks=c(0, 3, 6, 9, 12, 15, 18, 21))) + 
  add_quantile(y_value = 0.5, color = "gray50", linewidth = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0, 21, by = 3),
                     minor_breaks = seq(0, 21, 1)) +
  scale_color_manual(values = c('dark blue', 'red'),
                     labels = c('Control', 'Case')) +
  scale_fill_manual(values = c('dark blue', 'red'),
                    labels = c('Control', 'Case')) +
  theme(plot.title = element_text(face = "bold", family="serif", size = 14, hjust = 0.5)) +
  theme(legend.position = c(0.9, 0.9)) +
  theme(axis.title=element_text(size=10,  family="serif"), axis.text = element_text(size=10,  family="serif"), legend.text = element_text(size=10,  family="serif")) +
  theme(axis.title.x = element_blank())

### DIP

DIP <- readRDS("DIP_survival_170125.RDS")

# Use the Surv() function from the survival package to create a survival object 
s1_DIP <- Surv(DIP$time, DIP$outcome)

# Use the survfit() function and stratify by group
s1_DIP <- survfit(s1_DIP ~ DIP$group)

# Plot Kaplan-Meier survival curves for DIP cases and controls
plt_DIP <- survfit2(Surv(time, outcome) ~ group, data = DIP) %>% 
  ggsurvfit(linewidth = 1) +
  labs(
    y = "Percentage Survival",
    title = "Drug-induced parkinsonism",
    x = "Time from index date (years)"
  ) + 
  add_confidence_interval() +
  scale_ggsurvfit() + 
  add_quantile(y_value = 0.5, color = "gray50", linewidth = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0, 21, by = 3),
                     minor_breaks = seq(0, 21, 1)) +
  scale_color_manual(values = c('dark blue', 'red'),
                     labels = c('Control', 'Case')) +
  scale_fill_manual(values = c('dark blue', 'red'),
                    labels = c('Control', 'Case')) +
  theme(plot.title = element_text(face = "bold", family="serif", size = 14, hjust = 0.5)) +
  theme(legend.position = c(0.9, 0.9)) +
  theme(axis.title=element_text(size=10,  family="serif"), axis.text = element_text(size=10,  family="serif"), legend.text = element_text(size=10,  family="serif"))

### OSP

Secondary <- readRDS("OSP_new_survival_130325.RDS")

# Use the Surv() function from the survival package to create a survival object 
s1_Secondary <- Surv(Secondary$time, Secondary$outcome)

# Use the survfit() function and stratify by group
s1_Secondary <- survfit(s1_Secondary ~ Secondary$group)

# Plot Kaplan-Meier survival curves for secondary parkinsonism cases and controls
plt_Secondary <- survfit2(Surv(time, outcome) ~ group, data = Secondary) %>% 
  ggsurvfit(linewidth = 1) +
  labs(
    y = NULL,
    title = "Other secondary parkinsonism",
    x = "Time from index date (years)"
  ) + 
  add_confidence_interval() +
  scale_ggsurvfit() + 
  add_quantile(y_value = 0.5, color = "gray50", linewidth = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0, 21, by = 3),
                     minor_breaks = seq(0, 21, 1)) +
  scale_color_manual(values = c('dark blue', 'red'),
                     labels = c('Control', 'Case')) +
  scale_fill_manual(values = c('dark blue', 'red'),
                    labels = c('Control', 'Case')) +
  theme(plot.title = element_text(face = "bold", family="serif", size = 14, hjust = 0.5)) +
  theme(legend.position = c(0.9, 0.9)) +
  theme(axis.title=element_text(size=10,  family="serif"), axis.text = element_text(size=10,  family="serif"), legend.text = element_text(size=10,  family="serif"))

plt_Secondary

### Combine the eight KM curves into one composite figure
library("cowplot")

KM_8parkinsonisms_V1 <-plot_grid(plt_PD, plt_MSA, plt_PSP, plt_CBS, plt_DLB, plt_VP, plt_DIP, plt_Secondary, 
                                 labels = c("A", "B", "C", "D", "E", "F", "G", "H"),
                                 label_size = 11,
                                 label_fontfamily = "serif",
                                 ncol = 2, nrow = 4, align="hv", axis="b", greedy=FALSE)

# Save the KM plot as png, PDF and jpg files
ggsave(KM_8parkinsonisms_V1, 
       filename = "KM_parkinsonisms_300925.png",
       device = "png",
       bg="white",
       height = 8, width = 12, units = "in", dpi=600)

ggsave(KM_8parkinsonisms_V1, 
       filename = "KM_parkinsonisms_300925.pdf",
       device = "pdf",
       height = 8, width = 12, units = "in", dpi=600)

ggsave(KM_8parkinsonisms_V2, 
       filename = "KM_parkinsonisms_300925.jpg",
       device = "jpg",
       height = 8, width = 12, units = "in", dpi=600)

#################################################################################################################

### Age-stratified KM plots

# Read in PD dataset
PD <- readRDS("PD_survival_170125.RDS")

View(PD)

# Create a new column entitled age_group 
PD$age_group <- NA

PD2 <- PD %>%
  mutate(age_group = case_when(age_diagnosis < 65 ~ "< 65 years",
                               age_diagnosis >= 65 & age_diagnosis < 75 ~ "65-74 years",
                               age_diagnosis >= 75 & age_diagnosis < 85 ~ "75-84 years",
                               age_diagnosis>= 85 ~ "85 years and older"))
View(PD2)

# Filter age at diagnosis for cases < 65 years
Under65 <- PD2 %>%
  filter(age_group=="< 65 years")

# Use the Surv() function from the survival package to create a survival object for use 
# as the response in the model formula. There will be one entry for each subject that 
# is the survival time, which is followed by a + if the subject was censored. 
s1_PDunder65 <- Surv(Under65$time, Under65$outcome)

# Use the survfit() function and stratify by group (i.e. disease)
s1_PDunder65 <- survfit(s1_PDunder65 ~ Under65$group)

# Plot unadjusted Kaplan-Meier survival curves for PD cases and controls
plt_PD_under65 <- survfit2(Surv(time, outcome) ~ group, data = Under65) %>% 
  ggsurvfit(linewidth = 1) +
  labs(
    y = "Percentage Survival",
    title = "Cases less than 65 years at diagnosis",
    x = NULL
  ) + 
  add_confidence_interval() +
  scale_ggsurvfit() + 
  add_quantile(y_value = 0.5, color = "gray50", linewidth = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0, 21, by = 3),
                     minor_breaks = seq(0, 21, 1)) +
  scale_color_manual(values = c('dark blue', 'red'),
                     labels = c('Control', 'Case')) +
  scale_fill_manual(values = c('dark blue', 'red'),
                    labels = c('Control', 'Case')) +
  theme(plot.title = element_text(face = "bold", family="serif", size = 15, hjust = 0.5)) +
  theme(legend.position = c(0.9, 0.9)) +
  theme(axis.title=element_text(size=13,  family="serif"), axis.text = element_text(size=13,  family="serif"), legend.text = element_text(size=11,  family="serif"))

# Filter age at diagnosis to > 65 and <= 74

PD65_74 <- PD2 %>%
  filter(age_group=="65-74 years")
View(PD65_74)

s1_PD65_74 <- Surv(PD65_74$time, PD65_74$outcome)

# Use the survfit() function and stratify by group (i.e. disease)
s1_PD65_74 <- survfit(s1_PD65_74 ~ PD65_74$group)

# Plot unadjusted Kaplan-Meier survival curves for PD cases and controls
plt_PD65_74 <- survfit2(Surv(time, outcome) ~ group, data = PD65_74) %>% 
  ggsurvfit(linewidth = 1) +
  labs(
    y = "Percentage Survival",
    title = "Cases between 65 and 74 years at diagnosis",
    x = NULL
  ) + 
  add_confidence_interval() +
  scale_ggsurvfit() + 
  add_quantile(y_value = 0.5, color = "gray50", linewidth = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0, 21, by = 3),
                     minor_breaks = seq(0, 21, 1)) +
  scale_color_manual(values = c('dark blue', 'red'),
                     labels = c('Control', 'Case')) +
  scale_fill_manual(values = c('dark blue', 'red'),
                    labels = c('Control', 'Case')) +
  theme(plot.title = element_text(face = "bold", family="serif", size = 15, hjust = 0.5)) +
  theme(legend.position = c(0.9, 0.9)) +
  theme(axis.title=element_text(size=13,  family="serif"), axis.text = element_text(size=13,  family="serif"), legend.text = element_text(size=11,  family="serif"))

plt_PD65_74

# Filter age at diagnosis to 75 to 84 years

PD75_84 <- PD2 %>%
  filter(age_group=="75-84 years")

View(PD75_84)

s1_PD75_84 <- Surv(PD75_84$time, PD75_84$outcome)

# Use the survfit() function and stratify by group (i.e. disease)
s1_PD75_84 <- survfit(s1_PD75_84 ~ PD75_84$group)

# Plot unadjusted Kaplan-Meier survival curves for PD cases and controls
plt_PD75_84 <- survfit2(Surv(time, outcome) ~ group, data = PD75_84) %>% 
  ggsurvfit(linewidth = 1) +
  labs(
    y = "Percentage Survival",
    title = "Cases between 75 and 84 years at diagnosis",
    x = "Time from index date (years)"
  ) + 
  add_confidence_interval() +
  scale_ggsurvfit() + 
  add_quantile(y_value = 0.5, color = "gray50", linewidth = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0, 21, by = 3),
                     minor_breaks = seq(0, 21, 1)) +
  scale_color_manual(values = c('dark blue', 'red'),
                     labels = c('Control', 'Case')) +
  scale_fill_manual(values = c('dark blue', 'red'),
                    labels = c('Control', 'Case')) +
  theme(plot.title = element_text(face = "bold", family="serif", size = 15, hjust = 0.5)) +
  theme(legend.position = c(0.9, 0.9)) +
  theme(axis.title=element_text(size=13,  family="serif"), axis.text = element_text(size=13,  family="serif"), legend.text = element_text(size=11,  family="serif"))

####
# Filter cases to only show those 85 years and older

PDover85 <- PD2 %>%
  filter(age_group=="85 years and older")
View(PDover85)

s1_PDover85 <- Surv(PDover85$time, PDover85$outcome)

# Use the survfit() function and stratify by group (i.e. disease)
s1_PDover85 <- survfit(s1_PDover85 ~ PDover85$group)

# Plot unadjusted Kaplan-Meier survival curves for PD cases and controls
plt_PDover85 <- survfit2(Surv(time, outcome) ~ group, data = PDover85) %>% 
  ggsurvfit(linewidth = 1) +
  labs(
    y = "Percentage Survival",
    title = "Cases aged 85 years and older at diagnosis",
    x = "Time from index date (years)"
  ) + 
  add_confidence_interval() +
  scale_ggsurvfit() + 
  add_quantile(y_value = 0.5, color = "gray50", linewidth = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0, 21, by = 3),
                     minor_breaks = seq(0, 21, 1)) +
  scale_color_manual(values = c('dark blue', 'red'),
                     labels = c('Control', 'Case')) +
  scale_fill_manual(values = c('dark blue', 'red'),
                    labels = c('Control', 'Case')) +
  theme(plot.title = element_text(face = "bold", family="serif", size = 15, hjust = 0.5)) +
  theme(legend.position = c(0.9, 0.9)) +
  theme(axis.title=element_text(size=13,  family="serif"), axis.text = element_text(size=13,  family="serif"), legend.text = element_text(size=11,  family="serif"))


#### View the age-stratified KM plots
plt_PD_under65
plt_PD65_74
plt_PD75_84
plt_PDover85

# Combine the plots
library(cowplot)
KM_age2 <-plot_grid(plt_PD_under65, plt_PD65_74, plt_PD75_84, plt_PDover85, 
                    labels = c("A", "B", "C", "D"),
                    label_size = 11,
                    label_fontfamily = "serif",
                    ncol = 2, nrow = 2, align="hv", axis="b", greedy=FALSE)

# Save the age-stratified KM plots as png, PDF and jpg files
ggsave(KM_age2, 
       filename = "Age_stratified_KM_300925.png",
       device = "png",
       bg = "white",
       height = 8, width = 12, units = "in", dpi=600)

ggsave(KM_age2, 
       filename = "Age_stratified_KM_300925.pdf",
       device = "pdf",
       height = 8, width = 12, units = "in", dpi=600)

ggsave(KM_age2, 
       filename = "Age_stratified_KM_300925.jpg",
       device = "jpg",
       height = 8, width = 12, units = "in", dpi=600)

####################################################################

### MEDIAN AND PERCENTAGE SURVIVAL ESTIMATES FOR THE 8 PARKINSONIAN DISORDERS

### PD

# Produce a table of 5-year survival estimates for PD using the tbl_survfit() 
# function from the gtsummary package:
survfit(Surv(time, outcome) ~ group, data = PD) %>% 
  tbl_survfit(
    times = c(5, 10),
    estimate_fun = function(x) style_number(x, digits = 1, scale = 100),
    label_header = "**{time} year survival (95% CI)**"
  ) 

# Calculate median survival times
survfit(Surv(time, outcome) ~ group, data = PD) %>% 
  tbl_survfit(
    probs = 0.5,
    estimate_fun = function(x) style_number(x, digits = 1),
    label_header = "**Median survival (95% CI)**"
  )

### MSA

# 5 and 10 year survival for MSA
survfit(Surv(time, outcome) ~ group, data = MSA) %>% 
  tbl_survfit(
    times = c(5, 10),
    estimate_fun = function(x) style_number(x, digits = 1, scale = 100),
    label_header = "**{time} year survival (95% CI)**"
  ) 

# MSA median survival times
survfit(Surv(time, outcome) ~ group, data = MSA) %>% 
  tbl_survfit(
    probs = 0.5,
    estimate_fun = function(x) style_number(x, digits = 1),
    label_header = "**Median survival (95% CI)**")|>
  add_n()

### PSP 

# 5 and 10 year percentage survival for PSP
survfit(Surv(time, outcome) ~ group, data = PSP) %>% 
  tbl_survfit(
    times = c(5, 10),
    estimate_fun = function(x) style_number(x, digits = 1, scale = 100),
    label_header = "**{time} year survival (95% CI)**"
  ) 

# PSP median survival times
survfit(Surv(time, outcome) ~ group, data = PSP) %>% 
  tbl_survfit(
    probs = 0.5,
    estimate_fun = function(x) style_number(x, digits = 1),
    label_header = "**Median survival (95% CI)**")|>
  add_n()

### CBS

# 5 and 10 year percentage survival for CBS
survfit(Surv(time, outcome) ~ group, data = CBS) %>% 
  tbl_survfit(
    times = c(5, 10),
    estimate_fun = function(x) style_number(x, digits = 1, scale = 100),
    label_header = "**{time} year survival (95% CI)**"
  ) 

# CBS median survival times
survfit(Surv(time, outcome) ~ group, data = CBS) %>% 
  tbl_survfit(
    probs = 0.5,
    estimate_fun = function(x) style_number(x, digits = 1),
    label_header = "**Median survival (95% CI)**")|>
  add_n()

### DLB

# 5 and 10 year percentage survival for DLB
survfit(Surv(time, outcome) ~ group, data = DLB) %>% 
  tbl_survfit(
    times = c(5, 10),
    estimate_fun = function(x) style_number(x, digits = 1, scale = 100),
    label_header = "**{time} year survival (95% CI)**"
  ) 

# DLB median survival times
survfit(Surv(time, outcome) ~ group, data = DLB) %>% 
  tbl_survfit(
    probs = 0.5,
    estimate_fun = function(x) style_number(x, digits = 1),
    label_header = "**Median survival (95% CI)**")|>
  add_n()

### VP

# 5- and 10-year survival estimates for VP
survfit(Surv(time, outcome) ~ group, data = VP) %>% 
  tbl_survfit(
    times = c(5, 10),
    estimate_fun = function(x) style_number(x, digits = 1, scale = 100),
    label_header = "**{time} year survival (95% CI)**"
  ) 

# VP median survival times
survfit(Surv(time, outcome) ~ group, data = VP) %>% 
  tbl_survfit(
    probs = 0.5,
    estimate_fun = function(x) style_number(x, digits = 1),
    label_header = "**Median survival (95% CI)**")|>
  add_n()

### DIP

# 5- and 10-year survival estimates for DIP cases:
survfit(Surv(time, outcome) ~ group, data = DIP) %>% 
  tbl_survfit(
    times = c(5, 10),
    estimate_fun = function(x) style_number(x, digits = 1, scale = 100),
    label_header = "**{time} year survival (95% CI)**"
  ) 

# DIP median survival times
survfit(Surv(time, outcome) ~ group, data = DIP) %>% 
  tbl_survfit(
    probs = 0.5,
    estimate_fun = function(x) style_number(x, digits = 1),
    label_header = "**Median survival (95% CI)**")|>
  add_n()

### OSP 

# 5- and 10-year survival estimates for OSP:
survfit(Surv(time, outcome) ~ group, data = Secondary) %>% 
  tbl_survfit(
    times = c(5, 10),
    estimate_fun = function(x) style_number(x, digits = 1, scale = 100),
    label_header = "**{time} year survival (95% CI)**"
  ) 

# OSP median survival times:
survfit(Surv(time, outcome) ~ group, data = Secondary) %>% 
  tbl_survfit(
    probs = 0.5,
    estimate_fun = function(x) style_number(x, digits = 1),
    label_header = "**Median survival (95% CI)**")|>
  add_n()

################################################################

# POPULATION YEARS AT RISK AND NUMBERS OF DEATHS FOR EACH PARKINSONIAN DISORDER FOR LIFE EXPECTANCY CALCULATIONS

# Read in PD dataset
PD <- readRDS("PD_survival_170125.RDS")
View(PD)

# Turn off scientific notation
options(scipen=999)

### Extract the data required to populate the PHE life expectancy template spreadsheet

# Population years at risk for PD cases 

N.risk_PD <- PD %>%
  filter(group=="PD_patient") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_PD)

# Number of deaths for PD cases

N.deaths_PD <- PD %>%
  filter(group=="PD_patient" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population years at risk for PD controls

N.risk_controls <- PD %>%
  filter(group=="PD_control") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_controls)

# Number of deaths for PD controls

N.deaths.controls <- PD %>%
  filter(group=="PD_control" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

##### Gender breakdown of life expectancy for men and women 

# Population years at risk for MALE PD patients 

N.risk_male_PD <- PD %>%
  filter(group=="PD_patient" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

# Number of deaths for MALE PD patients
N.deaths_male_PD <- PD %>%
  filter(group=="PD_patient" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population years at risk for FEMALE PD patients 

N.risk_female_PD <- PD %>%
  filter(group=="PD_patient" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_female_PD)

# Number of deaths for FEMALE PD patients
N.deaths_female_PD <- PD %>%
  filter(group=="PD_patient" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

##### GENDER BREAKDOWN FOR CONTROLS

# Population years at risk for MALE PD controls 

N.risk_male_controls <- PD %>%
  filter(group=="PD_control" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_male_controls)

# Number of deaths for MALE PD controls
N.deaths_male_controls <- PD %>%
  filter(group=="PD_control" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population years at risk for FEMALE PD controls 

N.risk_female_controls <- PD %>%
  filter(group=="PD_control" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_female_controls)

# Number of deaths for FEMALE PD controls
N.deaths_female_control <- PD %>%
  filter(group=="PD_control" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

##################################################################

### PSP

PSP <- readRDS("PSP_new_survival_250225.RDS")
View(PSP)

# Population at risk for PSP cases
N.risk_PSP <- PSP %>%
  filter(group=="PSP_patient") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_PSP)

# Number of deaths for PSP cases

N.deaths_PSP <- PSP %>%
  filter(group=="PSP_patient" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for PSP controls
N.risk_PSP_controls <- PSP %>%
  filter(group=="PSP_control") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_PSP_controls)

# Number of deaths for PSP controls

N.deaths_PSP_controls <- PSP %>%
  filter(group=="PSP_control" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

#### PSP BY GENDER

# Population at risk for MALE PSP cases
N.risk_PSP_males <- PSP %>%
  filter(group=="PSP_patient" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_PSP_males)

# Number of deaths for MALE PSP cases

N.deaths_PSP_males <- PSP %>%
  filter(group=="PSP_patient" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE PSP cases
N.risk_PSP_females <- PSP %>%
  filter(group=="PSP_patient" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_PSP_females)

# Number of deaths for FEMALE PSP cases
N.deaths_PSP_females <- PSP %>%
  filter(group=="PSP_patient" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for MALE PSP controls
N.risk_PSP_control_males <- PSP %>%
  filter(group=="PSP_control" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_PSP_control_males)

# Number of deaths for MALE PSP controls

N.deaths_PSP_control_males <- PSP %>%
  filter(group=="PSP_control" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE PSP controls
N.risk_PSP_control_females <- PSP %>%
  filter(group=="PSP_control" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_PSP_control_females)

# Number of deaths for FEMALE PSP controls

N.deaths_PSP_control_females <- PSP %>%
  filter(group=="PSP_control" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

#############################################################################

# Read in updated MSA dataset 
MSA <- readRDS("MSA_new_survival_120325.RDS")

# Filter to show MSA cases
MSA_cases <- MSA %>%
  filter(case_type=="patient")
View(MSA_cases)

# Population at risk for MSA cases
N.risk_MSA <- MSA %>%
  filter(group=="MSA_patient") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_MSA)

# Number of deaths for MSA cases
N.deaths_MSA <- MSA %>%
  filter(group=="MSA_patient" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for MSA controls
N.risk_MSA_control <- MSA %>%
  filter(group=="MSA_control") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_MSA_control)

# Number of deaths for MSA controls
N.deaths_MSA_controls <- MSA %>%
  filter(group=="MSA_control" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

### MSA BY GENDER

# Population at risk for MALE MSA cases
N.risk_MSA_male <- MSA %>%
  filter(group=="MSA_patient" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_MSA_male)

# N. deaths for MALE MSA cases
N.deaths_MSA_male <- MSA %>%
  filter(group=="MSA_patient" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE MSA cases
N.risk_MSA_female <- MSA %>%
  filter(group=="MSA_patient" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_MSA_female)

# N. deaths for FEMALE MSA cases
N.deaths_MSA_female <- MSA %>%
  filter(group=="MSA_patient" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for MALE MSA controls
N.risk_MSA_control_male <- MSA %>%
  filter(group=="MSA_control" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_MSA_control_male)

# N. deaths for MALE MSA controls
N.deaths_MSA_control_male <- MSA %>%
  filter(group=="MSA_control" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE MSA controls
N.risk_MSA_control_female <- MSA %>%
  filter(group=="MSA_control" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_MSA_control_female)

# N. deaths for FEMALE MSA controls
N.deaths_MSA_control_female <- MSA %>%
  filter(group=="MSA_control" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

########################################################################################

### CBS

# Read in subsetted CBS data
CBS <- readRDS("CBS_survival_170125.RDS")

# Population at risk for CBS patients
N.risk_CBS <- CBS %>%
  filter(group=="CBS_patient") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_CBS)

# Number of deaths for CBS patients
N.deaths_CBS <- CBS %>%
  filter(group=="CBS_patient" & outcome=="1") %>%
  group_by(age_cat) %>%
  count

# Population at risk for CBS controls
N.risk_CBS_controls <- CBS %>%
  filter(group=="CBS_control") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_CBS_controls)

# Number of deaths for CBS controls 
N.deaths_CBS_controls <- CBS %>%
  filter(group=="CBS_control" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

### CBS BY GENDER

# Population at risk for MALE CBS patients
N.risk_CBS_male <- CBS %>%
  filter(group=="CBS_patient" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_CBS_male)

# Number of deaths for male CBS cases 
N.deaths_CBS_male <- CBS %>%
  filter(group=="CBS_patient" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE CBS patients
N.risk_CBS_female <- CBS %>%
  filter(group=="CBS_patient" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_CBS_female)

# Number of deaths for female CBS cases 
N.deaths_CBS_female <- CBS %>%
  filter(group=="CBS_patient" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for MALE CBS controls
N.risk_CBS_control_male <- CBS %>%
  filter(group=="CBS_control" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_CBS_control_male)

# Number of deaths for male CBS controls 
N.deaths_CBS_control_male <- CBS %>%
  filter(group=="CBS_control" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE CBS controls
N.risk_CBS_control_female <- CBS %>%
  filter(group=="CBS_control" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))
View(N.risk_CBS_control_female)

# Number of deaths for female CBS controls 
N.deaths_CBS_control_female <- CBS %>%
  filter(group=="CBS_control" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

################################################################################

### DLB

DLB_new <- readRDS("DLB_new_survival_CPRDonly_240225.RDS")

# Population at risk for DLB cases 
N.risk_DLB <- DLB_new %>%
  filter(group=="DLB_patient") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

# Number of deaths for DLB cases
N.deaths_DLB_cases <- DLB_new %>%
  filter(group=="DLB_patient" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for DLB controls
N.risk_DLB_control <- DLB_new %>%
  filter(group=="DLB_control") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_DLB_control)

# Number of deaths for DLB controls
N.deaths_DLB_controls <- DLB_new %>%
  filter(group=="DLB_control" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

### DLB BY GENDER

# Population at risk for MALE DLB cases 
N.risk_DLB_male <- DLB_new %>%
  filter(group=="DLB_patient" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_DLB_male)

# Number of deaths for MALE DLB cases
N.deaths_DLB_male <- DLB_new %>%
  filter(group=="DLB_patient" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE DLB cases 
N.risk_DLB_female <- DLB_new %>%
  filter(group=="DLB_patient" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_DLB_female)

# Number of deaths for FEMALE DLB cases
N.deaths_DLB_female <- DLB_new %>%
  filter(group=="DLB_patient" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for MALE DLB controls 
N.risk_DLB_control_male <- DLB_new %>%
  filter(group=="DLB_control" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_DLB_control_male)

# Number of deaths for MALE DLB controls
N.deaths_DLB_control_male <- DLB_new %>%
  filter(group=="DLB_control" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE DLB controls 
N.risk_DLB_control_female <- DLB_new %>%
  filter(group=="DLB_control" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_DLB_control_female)

# Number of deaths for FEMALE DLB controls
N.deaths_DLB_control_female <- DLB_new %>%
  filter(group=="DLB_control" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

############################################################################

VP <- readRDS("VP_new_survival_130325.RDS")

# Population at risk for VP cases
N.risk_VP <- VP %>%
  filter(group=="VP_patient") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_VP)

# Number of deaths for VP cases
N.deaths_VP <- VP %>%
  filter(group=="VP_patient" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for VP controls
N.risk_VP_control <- VP %>%
  filter(group=="VP_control") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_VP_control)

# Number of deaths for VP controls
N.deaths_VP_controls <- VP %>%
  filter(group=="VP_control" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

### VP BY GENDER

# Population at risk for MALE VP cases
N.risk_VP_male <- VP %>%
  filter(group=="VP_patient" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_VP_male)

# Number of deaths for MALE VP cases
N.deaths_VP_male <- VP %>%
  filter(group=="VP_patient" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE VP cases
N.risk_VP_female <- VP %>%
  filter(group=="VP_patient" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_VP_female)

# Number of deaths for FEMALE VP cases
N.deaths_VP_female <- VP %>%
  filter(group=="VP_patient" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for MALE VP controls
N.risk_VP_control_male <- VP %>%
  filter(group=="VP_control" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_VP_control_male)

# Number of deaths for MALE VP controls
N.deaths_VP_control_male <- VP %>%
  filter(group=="VP_control" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE VP controls
N.risk_VP_control_female <- VP %>%
  filter(group=="VP_control" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_VP_control_female)

# Number of deaths for FEMALE VP controls
N.deaths_VP_control_female <- VP %>%
  filter(group=="VP_control" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

############################################################################################

### DIP

DIP <- readRDS("DIP_survival_170125.RDS")

# Population at risk for DIP cases
N.risk_DIP <- DIP %>%
  filter(group=="DIP_patient") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_DIP)

# Number of deaths for DIP cases
N.deaths_DIP <- DIP %>%
  filter(group=="DIP_patient" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for DIP controls
N.risk_DIP_controls <- DIP %>%
  filter(group=="DIP_control") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_DIP_controls)

# Number of deaths for DIP controls
N.deaths_DIP_controls <- DIP %>%
  filter(group=="DIP_control" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

### DIP BY GENDER

# Population at risk for MALE DIP cases
N.risk_DIP_male <- DIP %>%
  filter(group=="DIP_patient" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_DIP_male)

# Number of deaths for MALE DIP cases
N.deaths_DIP_male <- DIP %>%
  filter(group=="DIP_patient" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE DIP cases
N.risk_DIP_female <- DIP %>%
  filter(group=="DIP_patient" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_DIP_female)

# Number of deaths for FEMALE DIP cases
N.deaths_DIP_female <- DIP %>%
  filter(group=="DIP_patient" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for MALE DIP controls
N.risk_DIP_control_male <- DIP %>%
  filter(group=="DIP_control" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_DIP_control_male)

# Number of deaths for MALE DIP controls
N.deaths_DIP_controls_male <- DIP %>%
  filter(group=="DIP_control" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE DIP controls
N.risk_DIP_control_female <- DIP %>%
  filter(group=="DIP_control" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_DIP_control_female)

# Number of deaths for FEMALE DIP controls
N.deaths_DIP_controls_female <- DIP %>%
  filter(group=="DIP_control" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

###########################################################################

# Read in updated OSP data
OSP <- readRDS("OSP_new_survival_130325.RDS")

# Population at risk for OSP cases
N.risk_OSP <- OSP %>%
  filter(group=="Secondary_patient") %>%
  group_by(age_cat) %>%
  summarise(sum(time))
View(N.risk_OSP)

# N. deaths for OSP cases
N.deaths_OSP <- OSP %>%
  filter(group=="Secondary_patient" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for OSP controls
N.risk_OSP_controls <- OSP %>%
  filter(group=="Secondary_control") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_OSP_controls)

# N. deaths for OSP controls
N.deaths_OSP_controls <- OSP %>%
  filter(group=="Secondary_control" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

### OSP BY GENDER

# Population at risk for MALE OSP cases
N.risk_OSP_male <- OSP %>%
  filter(group=="Secondary_patient" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_OSP_male)

# N. deaths for MALE OSP cases
N.deaths_OSP_male <- OSP %>%
  filter(group=="Secondary_patient" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE OSP cases
N.risk_OSP_female <- OSP %>%
  filter(group=="Secondary_patient" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_OSP_female)

# N. deaths for FEMALE OSP cases
N.deaths_OSP_female <- OSP %>%
  filter(group=="Secondary_patient" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for MALE OSP controls
N.risk_OSP_control_male <- OSP %>%
  filter(group=="Secondary_control" & gender=="Male") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_OSP_control_male)

# N. deaths for MALE OSP controls
N.deaths_OSP_control_male <- OSP %>%
  filter(group=="Secondary_control" & gender=="Male" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

# Population at risk for FEMALE OSP controls
N.risk_OSP_control_female <- OSP %>%
  filter(group=="Secondary_control" & gender=="Female") %>%
  group_by(age_cat) %>%
  summarise(sum(time))

View(N.risk_OSP_control_female)

# N. deaths for FEMALE OSP controls
N.deaths_OSP_control_female <- OSP %>%
  filter(group=="Secondary_control" & gender=="Female" & outcome=="1") %>%
  group_by(age_cat) %>%
  count()

###############################################################################################################

### COX MODELLING

# Set working directory
setwd("C:/Users/2877245G/OneDrive - University of Glasgow/Documents/MD/Incidence and Prevalence/Analysis/Survival/CPRD_HES data/Survival Analysis")

# Read in PD dataset
PD <- readRDS("PD_survival_170125.RDS")
View(PD)

# Turn off scientific notation
options(scipen=999)

# Edit the PD dataset to only include variables required for modelling
names(PD)

PD <- PD[, c(1, 3, 8:10, 14:16, 31:32)]

# Create a new column entitled 'PD_case'. Assign cases 1 if they are a patient and 0 if they are a control. 
PD$PD_case <- NA

PD <- PD %>% 
  mutate(PD_case = case_when(group=="PD_patient" ~ 1,
                             group=="PD_control" ~ 0))

# Create new column entitled 'male gender' and assign cases 1 if they are male and 0 if they are female
PD$male_gender <- NA

PD <- PD %>%
  mutate(male_gender = case_when(gender=="Male" ~ 1,
                                 gender=="Female" ~ 0))

View(PD)

# Create a new column entitled age_group 
PD$age_group <- NA

PD2 <- PD %>%
  mutate(age_group = case_when(age_diagnosis < 65 ~ "< 65 years",
                               age_diagnosis >= 65 & age_diagnosis < 75 ~ "65-74 years",
                               age_diagnosis >= 75 & age_diagnosis < 85 ~ "75-84 years",
                               age_diagnosis>= 85 ~ "85 years and older"))

# Save updated modelling dataset as PD_cox
saveRDS(PD2, "PD_cox_dataset_280125.RDS")

##############################

### MODELLING: PD VS CONTROLS

# Read in updated dataset
PD <- readRDS("PD_cox_dataset_280125.RDS")

View(PD)

# Fit a simple Cox regression model (model 1a) where outcome is predicted by the covariates: group, age_group and gender
model1a <- coxph(Surv(time, outcome) ~ PD_case + age_group + male_gender, data=PD)
summary(model1a)

# Produce a forest plot of model 1a
ggforest(model1a, data=as.data.frame(PD))

# Test the CPH assumption
model1a_ph <- cox.zph(model1a)

# Plot Schoenfeld residuals to assess the CPH assumption
ggcoxzph(model1a_ph)

# Plot the log-log curves for group
loglog_group <- survfit2(Surv(time, outcome) ~ group, data = PD)
plot(loglog_group, fun = "cloglog", xlab = "Time in years using log",
     ylab = "log-log survival", main = "log-log curves by PD group",
     col=c("red", "darkblue"))
legend("topleft", inset=0.02, legend=c("PD case", "PD control"),
       col=c("red", "darkblue"), lty=1, cex=0.8, bty = "n")

# Plot log log curves for age_category 
loglog_agegroup <- survfit2(Surv(time, outcome) ~ age_group, data = PD)
plot(loglog_agegroup, fun = "cloglog", xlab = "Time in years using log",
     ylab = "log-log survival", main = "log-log curves")

# Plot log log curves for gender
loglog_gender <- survfit2(Surv(time, outcome) ~ gender, data = PD)
plot(loglog_gender, fun = "cloglog", xlab = "Time in years using log",
     ylab = "log-log survival", main = "log-log curves",
     col=c("black", "darkblue"))

### Time-partitioned extended Cox models 

# STEP 1: partitioning time to address NPH as it appears that the CPH assumption is not upheld for age_group and group.
PD_split <- survSplit(PD, cut=c(7, 14), end="time", event="outcome", zero=0, episode="tgroup")
View(PD_split)

# Save this new dataset
saveRDS(PD_split, "PD_Cox_timesplit_280125.Rds")

##########

PD2 <- readRDS("PD_Cox_timesplit_280125.Rds")

# Fit a model for PD2 entitled model 2 where covariates include male gender, PD_case and age_group 
# with the last 2 covariates partitioned by time period
model2 <- coxph(Surv(tstart, time, outcome) ~ male_gender + PD_case + age_group:strata(tgroup), data=PD2)

# Check the PH assumption for model 2
model2_ph <- cox.zph(model2)
ggcoxzph(model2_ph)

# Produce forest plots for model 2
ggforest(model2, data=as.data.frame(PD2))

# Compare the fit of model 2 vs. model 1a
extractAIC(model1a); extractAIC(model2)

### 

# Fit a model entitled model 3 where covariates include male_gender, PD_case and age_group with both PD_case and age_group 
# partitioned by time period
model3 <- coxph(Surv(tstart, time, outcome) ~ male_gender + PD_case:strata(tgroup) + age_group:strata(tgroup), data=PD2)

summary(model3)

# Produce forest plots for model 3
ggforest(model3, data=as.data.frame(PD2))

# Compare the fit of model 2 vs. model 3
extractAIC(model2); extractAIC(model3)
