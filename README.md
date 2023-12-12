# HuMiLity - Modeling Semantic Plausibility

This repository contains the code for data analysis as well as code for modeling semantic plausibility in winter 2023/2024 by Huirong Tan, Miriam Segiet, and Li Lin.

First check requirements.txt to see if any packages still need to be installed in your environment.

Next, you may see folder Data_analysis in which our code for dataset analysis of 12 characteristics on pap and pep-3k are stored: 

### Data_analysis folder contains these files:
* data_analysis.py: for pep-3k dataset analysis. The analysis mainly includes:

1. Overall Statistics and Word Counts
2. Word and Phrase Statistics
3. Word Frequency
4. Bigram POS-Tag Counts
5. Sentence Readability

* data_analysis_liln.py: for pap and pep-3k dataset analysis.
* data_analysis_pap_abstractness.py: for abstractness analysis of pap dataset.
* demo_display.ipynb: display of partial analysis results on pap and pep-3k datasets. The analysis mainly includes:
1. Basic information: number of data and binary classes.
2. Tokens: number of total tokens, number of unique tokens, tokens pair
3. Pos analysis: number of unigram pos, number of bigram pos
4. Abstractness: unigram abstractness tag, bigram abstractness, tokens and abstractness

* pap analysis_Huirong.ipynb: for pap analysis, mainly includes:
5.  Word Frequency Distribution
6. Annotation Difference

