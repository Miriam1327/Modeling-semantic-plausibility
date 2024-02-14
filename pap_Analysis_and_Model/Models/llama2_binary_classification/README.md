**This folder contains one folder and the following files:**
* llama2_Model.py: Get the peft llama2-7b model and perform 8bit Quantization on the model.
* pap_Dataloader.py: Process the pap data file and return a dataloader containing tokenized text and 0,1 labels.
* evaluation.py: Get accuracy, precision, recall and f-score.
* pap_llama2_Train.ipynb: The code for model fine-tuning using the pap training dataset, and the display of semantic plausibility binary classification results for the pap test dataset.
* pep3k_llama2_Train.ipynb: The code for model fine-tuning using the pep3k training dataset, and the display of semantic plausibility binary classification results for the pep3k test dataset.
* pap&pep3k_llama2_Train.ipynb: The code for model fine-tuning using the pap+pep3k training dataset, and the display of semantic plausibility binary classification results for the pap and pep3k test datasets.
* requirements.txt
* Readme.md
* `result_analysis` folder contains these files:
    
    * pap_preds_filter.txt: classified results by the first pap fine-tuned llama2-7b model.
    * pap_preds_filter1.txt: classified results by the second pap fine-tuned llama2-7b model.
    * pap_preds.txt: classified results by pap+pep3k fine-tuned llama2-7b model.
    * data_wrong_analysis.py: for model misclassified results analysis.
    * llama2_analysis.ipynb: display of model results analysis.


**To run pap_llama2_train.ipynb/pep3k_llama2_Train.ipynb/pap&pep3k_llama2_Train.ipynb  code, the following operations are required:**
1. Install environment required libraries:
    1. option 1:
        * pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes
        * pip install sentencepiece
        * pip install trl 
        * pip install matplotlib
    2. option 2:
        * pip install -r requirements.txt

2. Add data files to this location:
    * Modeling_Semantic_Plausibility/Data/pap

