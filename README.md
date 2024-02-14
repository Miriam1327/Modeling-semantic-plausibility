# HuMiLity - Modeling Semantic Plausibility

This repository contains the code for data analysis as well as code for modeling semantic plausibility in winter 2023/2024 by Huirong Tan, Miriam Segiet, and Li Lin.

> When you need to setup a new environment, use `pip install -r requirements.txt` to load the required packages stored in the corresponding file. If not already satisfied, make sure to download the tagger part of the nltk library by using `nltk.download('averaged_perceptron_tagger')`.

The whole analysis and classification based on the PEP-3K dataset is included in the folder `pep-3k_classification`. 
This folder contains five separate files:
* pep3k_data_analysis.py which contains the analysis of the PEP-3K dataset
* pep3k_keras.py which contains a classification model based on keras and additional training data from PAP
* pep3k_evaluation.py which contains the evaluation based on precision, recall, f1 score, and roc auc scores
* pep3k_keras_data_prep.py which is similar to pep3k_keras.py but adapts the PEP-3K and PAP files according to given rules
* pep3k_evaluation_data_prep.py which contains the evaluation based on precision, recall, f1 score, and roc auc scores given the data adaption 

The `pap_Analysis_and_Model` folder mainly includes data analysis of the PAP dataset and the implementation of two semantic plausibility classification models, as well as part of the pep3k dataset analysis and model classification.
This folder contains two separate folders:
* `Data_analysis` folder contains these files:

    * data_analysis_compare.py: for pap and pep-3k dataset analysis.
    * data_analysis_pap_abstractness.py: for abstractness analysis of pap dataset.
    * demo_display.ipynb: the display of partial analysis results on pap and pep-3k datasets. 
    * pap analysis_WA.ipynb: for pap analysis of Word Frequency Distribution and Annotation Difference.

* `Models` folder contains two models:
    * pap_roberta folder:
        * pap_roberta.ipynb: model training based on Roberta with pap and extra training data from pep3k; model evaluation.
    * llama2_8bit_peft floder:
        * see the Readme.md in this floder.




