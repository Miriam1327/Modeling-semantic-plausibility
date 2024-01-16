# HuMiLity - Modeling Semantic Plausibility

This repository contains the code for data analysis as well as code for modeling semantic plausibility in winter 2023/2024 by Huirong Tan, Miriam Segiet, and Li Lin.

> When you need to setup a new environment, use `pip install -m requirements.txt` to load the required packages stored in the corresponding file. If not already satisfied, make sure to download the tagger part of the nltk library by using `nltk.download('averaged_perceptron_tagger')`.

The whole analysis and classification based on the pep-3k dataset is included in the folder `pep-3k_classification`. 
This folder contains three separate files:
* pep3k_data_analysis.py which contains the analysis of the pap-3k dataset
* pep3k_keras.py which contains a classification model based on keras and additional training data from pap
* pep3k_evaluation.py which contains the evaluation based on precision, recall, f1 score, and roc auc scores

Next, you may see folder Data_analysis in which our code for dataset analysis of characteristics on pap and pep-3k are stored: 

### Data_analysis folder contains these files:
* data_analysis.py: for pep-3k dataset analysis. 

The analysis mainly includes:

    1. Overall Statistics and Word Counts
    2. Word and Phrase Statistics
    3. Word Frequency
    4. Bigram POS-Tag Counts
    5. Sentence Readability


* data_analysis_liln.py: for pap and pep-3k dataset analysis.
* data_analysis_pap_abstractness.py: for abstractness analysis of pap dataset.
* demo_display.ipynb: the display of partial analysis results on pap and pep-3k datasets. To run the code in it, the data location is in: Modeling_Semantic_Plausibility/Data/pap, Modeling_Semantic_Plausibility/Data/pep-3k. 

The analysis mainly includes:

    1. Basic information: number of data and binary classes
    2. Tokens: number of total tokens, number of unique tokens, tokens pair
    3. Pos analysis: number of unigram pos, number of bigram pos
    4. Abstractness: unigram abstractness tag, bigram abstractness, tokens and abstractness


* pap analysis_Huirong.ipynb: for pap analysis, mainly includes:

The analysis mainly includes:

    5.  Word Frequency Distribution
    6. Annotation Difference

