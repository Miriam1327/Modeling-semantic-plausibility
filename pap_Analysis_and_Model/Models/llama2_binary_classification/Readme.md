This folder contains the following files:
* llama2_Model.py: Get the peft llama2-7b model and perform 8bit Quantization on the model.
* pap_Dataloader.py: Process the pap data file and return a dataloader containing tokenized text and 0,1 labels.
* evaluation.py: Get accuracy, precision, recall and f-score.
* pap_llama2_Train.ipynb: Train the model and get display of binary classification inference results.
* requirements.txt
* Readme.md


To run pap_llama2_train.ipynb code, the following operations are required: 

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

