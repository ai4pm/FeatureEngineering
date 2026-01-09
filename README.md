# FeatureEngineeringComparison

This repository contains all the necessary files to reproduce the results of the feature engineering comparison study.

Codes.zip - Contains all the Python files required to reproduce the results. Extract this folder to access the code.

AutoencoderModels.zip, PCA_data.zip - Contains autoencoder models and PCA data. Extract this folder and place it in the root directory where the main Python files are located.

Genetic_Ancestry.xlsx - An Excel file containing ethnic group information associated with each sample and cancer type.

MachineLearningTasks.xlsx - An Excel file detailing the number of samples (positive and negative classes) for data-abundant (source domain) and data-disadvantaged (target domain) groups associated with 908 machine learning tasks.

SingleOmics_AE_Linear-Linear-MSE.zip - Contains the results of 908 machine learning tasks using Autoencoder based feature extraction for the first setting (AE-1), where both f_1(x) and f_2(x) are linear activation functions, and the loss function L is a mean squared error (MSE).

Summary_SingleOmics_AE_Linear-Linear-MSE.xlsx - An Excel file summarizing the results of 908 machine learning tasks using Autoencoder based feature extraction for the first setting (AE-1).

SingleOmics_AE_ReLU-Sigmoid-BCE.zip - Contains the results of 908 machine learning tasks using Autoencoder based feature extraction for the second setting (AE-2), where f_1(x) is a Rectified Linear Unit (ReLU) activation function, f_2(x) is a Sigmoid activation function, and the loss function L is binary cross-entropy (BCE).

Summary_SingleOmics_AE_ReLU-Sigmoid-BCE.xlsx - An Excel file summarizing the results of 908 machine learning tasks using Autoencoder based feature extraction for the second setting (AE-2).

SingleOmics_ANOVA.zip - Contains the results of 908 machine learning tasks using the ANOVA based feature selection method.

Summary_SingleOmics_ANOVA.xlsx - An Excel file summarizing the results of 908 machine learning tasks using the ANOVA based feature selection method.

SingleOmics_PCA.zip - Contains the results of 908 machine learning tasks using the PCA based dimensionality reduction method.

Summary_SingleOmics_PCA.xlsx - An Excel file summarizing the results of 908 machine learning tasks using the PCA based dimensionality reduction method.

Supplementary_Information.pdf - A document containing the abbreviations of 33 cancer types from the TCGA dataset and 7 Pan-cancers using combinations of cancer types.

TCGA-CDR-SupplementalTableS1.xlsx - An Excel file providing clinical outcome endpoints and event time threshold information associated with each sample and cancer type.

TasksCount.xlsx - An Excel file listing the number of machine learning tasks assembled considering various omics features, target groups, and clinical outcome endpoints for different feature engineering methods: ANOVA, PCA, AE-1 (AE-L-L-MSE), and AE-2 (AE-R-S-BCE).

Sample_Label_Expression_Info - This folder consists of sample and label information and the expression values for 99 datasets (33 cancer types X 3 omics features, i.e., mRNA-Expression, DNA Methylation, and MicroRNA Expression) for four clinical outcome endpoints, i.e., OS, DSS, PFI, and DFI. 


**Codes.zip**

Extract Codes.zip and place the contents in the root directory.

`main.py` - This is the main Python script to be executed. It processes the data and performs the feature engineering comparisons.

`main.sh` - This is the shell script to run the main Python script using SLURM for job scheduling. Ensure SLURM is configured in your environment.


**Prerequisites Installation**

Python 3.7 - https://www.python.org/downloads/release/python-370/

Numpy 1.21.5 - https://pypi.org/project/numpy/1.21.5/

Pandas 1.3.5 - https://pypi.org/project/pandas/1.3.5/

Scipy 1.7.3 - https://pypi.org/project/scipy/1.7.3/

Scikit-learn 1.0.2 - https://pypi.org/project/scikit-learn/1.0.2/

Theano 1.0.3 - https://pypi.org/project/Theano/1.0.3/

Tensorflow 1.13.1 - https://pypi.org/project/tensorflow/1.13.1/

Tensorflow-estimator 1.13.0 - https://pypi.org/project/tensorflow-estimator/1.13.0/

Tensorboard 1.13.1 - https://pypi.org/project/tensorboard/1.13.1/

Keras 2.2.4 - https://pypi.org/project/keras/2.2.4/

Keras-applications 1.0.8 - https://pypi.org/project/Keras-Applications/

Keras-preprocessing 1.1.0 - https://pypi.org/project/Keras-Preprocessing/1.1.0/

Pytorch 1.10.2 - https://pypi.org/project/torch/1.10.2/

Lasagne 0.2.dev1 - https://github.com/Lasagne/Lasagne

Xlrd 1.1.0 - https://pypi.org/project/xlrd/1.1.0/

openpyxl - https://pypi.org/project/openpyxl/


**Running the codes**

STEP 1 - Download the required dataset as per the instructions provided here: https://github.com/ai4pm/MultiEthnicMachineLearning

STEP 2 - Run `main.py` with input arguments using the following command:

```sh
python main.py <cancer_type> <omics_feature_type> <clinical_outcome_endpoint> <event_time_threshold> <target_DDP_group>
```

After execution, the result will be saved in the `Result` folder under the `Codes` folder as an Excel file.

STEP 3 - For different feature engineering methods, the following commands should be modified in the `main.py` file.

For ANOVA based feature selection method, set the following: 

```sh
AE=False and PCA_FE_All=False
```

For PCA based dimensionality reduction method, set the following: 

```sh
AE=False and PCA_FE_All=True
```

For AE-1 based feature extraction method, set the following: 

```sh
AE=True
EncoderActivation = 'linear'
DecoderActivation = 'linear'
LossFunction = 'mean_squared_error'
PCA_FE_All=False
```

For AE-2 based feature extraction method, set the following: 

```sh
AE=True
EncoderActivation = 'relu'
DecoderActivation = 'sigmoid'
LossFunction = 'binary_crossentropy'
PCA_FE_All=False
```


## Acknowledgement

This work has been supported by NIH R01 grant.


## Contact

Dr. Teena Sharma (tee.shar6@gmail.com)

Prof. Yan Cui (ycui2@uthsc.edu)


## Reference

T. Sharma, N. K. Verma, and Y. Cui, "Multiethnic Machine Learning for Omics based Cancer Prognosis: From Feature Engineering to Disparity Detection and Mitigation," IEEE Transactions on Artificial Intelligence, pp. 1-12, Dec. 2025. DOI: 10.1109/TAI.2025.3640596 (Early Access)


