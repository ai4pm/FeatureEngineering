# FeatureEngineeringComparison

This repository contains all the necessary files to reproduce the results of the feature engineering comparison study.

Codes.zip - Contains all the Python files required to reproduce the results. Extract this folder to access the code.

AutoencoderModels.zip, PCA_data.zip - Contains autoencoder models and PCA data. Extract this folder and place it in the root directory where the main Python files are located.

Genetic_Ancestry.xlsx - An Excel file containing ethnic group information associated with each sample and cancer type.

MachineLearningTasks.xlsx - An Excel file detailing the number of samples (positive and negative classes) for data-abundant (source domain) and data-disadvantaged (target domain) groups associated with 908 machine learning tasks.

SingleOmics_AE_Linear-Linear-MSE.zip - Contains the results of 908 machine learning tasks using Autoencoder-based feature extraction for the first setting (AE-1), where both f_1(x) and f_2(x) are linear activation functions, and the loss function L is a mean squared error (MSE).

Summary_SingleOmics_AE_Linear-Linear-MSE.xlsx - An Excel file summarizing the results of 908 machine learning tasks using Autoencoder-based feature extraction for the first setting (AE-1).

SingleOmics_AE_ReLU-Sigmoid-BCE.zip - Contains the results of 908 machine learning tasks using Autoencoder-based feature extraction for the second setting (AE-2), where f_1(x) is a Rectified Linear Unit (ReLU) activation function, f_2(x) is a linear activation function, and the loss function L is binary cross-entropy (BCE).

Summary_SingleOmics_AE_ReLU-Sigmoid-BCE.xlsx - An Excel file summarizing the results of 908 machine learning tasks using Autoencoder-based feature extraction for the second setting (AE-2).

SingleOmics_ANOVA.zip - Contains the results of 908 machine learning tasks using the ANOVA-based feature selection method.

Summary_SingleOmics_ANOVA.xlsx - An Excel file summarizing the results of 908 machine learning tasks using the ANOVA-based feature selection method.

SingleOmics_PCA.zip - Contains the results of 908 machine learning tasks using the PCA-based dimensionality reduction method.

Summary_SingleOmics_PCA.xlsx - An Excel file summarizing the results of 908 machine learning tasks using the PCA-based dimensionality reduction method.

Supplementary_Information.pdf - A document containing the abbreviations of 33 cancer types from the TCGA dataset and 7 Pan-cancers using combinations of cancer types.

TCGA-CDR-SupplementalTableS1.xlsx - An Excel file providing clinical outcome endpoints and event time thresholds information associated with each sample and cancer type.

TasksCount.xlsx - An Excel file listing the number of machine learning tasks assembled considering various omics features, target groups, and clinical outcome endpoints for different feature engineering methods: ANOVA, PCA, AE-1 (AE-L-L-MSE), and AE-2 (AE-R-S-BCE).



