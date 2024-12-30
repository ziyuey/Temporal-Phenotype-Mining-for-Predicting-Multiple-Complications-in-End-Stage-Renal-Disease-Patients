# Temporal-Phenotype-Mining-for-Predicting-Multiple-Complications-in-End-Stage-Renal-Disease-Patients

This repository contains the source code for the paper **"Temporal Phenotype Mining for Predicting Multiple Complications in End-Stage Renal Disease Patients"**. The paper proposes a method to predict multiple complications in patients with end-stage renal disease (ESRD) by mining temporal phenotypes from longitudinal clinical data.

## Overview

End-stage renal disease (ESRD) patients are at risk for various complications. Predicting these complications early is critical for improving patient care and outcomes. This project introduces an approach that integrates temporal data analysis and phenotype mining to predict the onset of multiple complications in ESRD patients.

Our model builds upon the **[CAMELOT](https://github.com/hrna-ox/camelot-icml)** framework, and we thank the authors of CAMELOT for their foundational work. We extend the CAMELOT framework to address the specific challenges faced by patients with ESRD. Our work introduces several key advancements:

- **Temporal Phenotyping**: We incorporate temporal phenotyping to improve the prediction of multiple comorbidities and enhance interpretability.
- **Survival Analysis**: We introduce survival analysis as an auxiliary task to better understand patient outcomes over time.
- **Outcome-Sensitive Clusters**: We provide a detailed examination of how outcome-sensitive clusters evolve over time within patient trajectories.

This extension of the CAMELOT framework allows for a more nuanced understanding of disease progression and supports more precise clinical decision-making in the management of patients with ESRD.



