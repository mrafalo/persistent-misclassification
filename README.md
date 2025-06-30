# Persistent Misclassification in Thyroid Ultrasound Imaging

This repository contains the code, data preparation scripts, and experimental results for the study:

**"Persistent Misclassification as a Source of Diagnostic Uncertainty in Thyroid Ultrasound Imaging"**

## Overview

We present a novel approach for identifying **persistently misclassified images** in real-world thyroid ultrasound datasets. Using a set of **484 thyroid nodule images**, we evaluated **four convolutional neural network (CNN) architectures** to uncover images that are repeatedly misclassified across models and cross-validation folds.

Key contributions of this study:

- **Persistent Misclassification**: Defined as images consistently misclassified regardless of model architecture or training split.
- **Radiological Validation**: An expert radiologist reviews misclassified cases for clinical ambiguity or atypical features.
- **Grad-CAM Analysis**: Used to visualize the regions influencing model decisions for both correct and incorrect predictions.
- **Clinical Relevance**: Highlights how model-independent diagnostic uncertainty can be rooted in data ambiguity rather than architecture design.
- **Call for Data Validation**: We argue for systematic integration of **data quality checks** alongside accuracy metrics in AI model development pipelines.
