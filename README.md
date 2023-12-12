# HuMiLity - Modeling Semantic Plausibility

This repository contains the code for data analysis as well as code for modeling semantic plausibility in winter 2023/2024 by Huirong Tan, Miriam Segiet, and Li Lin.

First check requirements.txt to see if any packages still need to be installed in your environment.

Next, you may see folder Data_analysis in which our code for dataset analysis of 12 characteristics on pap and pep-3k are stored: 

Pep-3k
1. Overall Statistics and Word Counts
2. Word and Phrase Statistics
3. Word Frequency
4. Bigram POS-Tag Counts
5. Sentence Readability
6. Find s-v, v-o pair

Pap 
7. Basic information: number of data, the proportion of classes.
8. Tokens: number of total tokens, number of unique tokens(Overall Statistics and Word Counts)
9. Word Frequency Distribution
10. Annotation Difference 
11. Pos: number of unique pos, distribution of unique pos, distribution of bigram pos
12. Abstractness

Check Data_analysis/data_analysis.py to see code for characteristics 1-5 from Miri, Data_analysis/data_analysis_liln.py for 6-8 and 11-12 from Li, and pap_analysis_Huirong.ipynb for 9-10 from Huirong. Also, Data_analysis/demo_display.ipynb is there for visualize our results.
