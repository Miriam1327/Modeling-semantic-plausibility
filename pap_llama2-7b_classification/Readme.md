This folder contains the following files:
* llama2_Model.py: Get the Peftmodel and perform 8bit Quantization on the model.
* pap_Dataloader.py: Process the pap data file and return a dataloader containing tokenized text and 0,1 labels.
* evaluation.py: Get accuracy, precision, recall and f-score.
* pap_llama2_Train.ipynb: Train the model and get classified inference results.


To run pap_llama2_train.ipynb code, the following operations are required: 

1. Environment installation:
    1. method 1:
        * ! pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes
        * ! pip install sentencepiece
        * ! pip install  trl 
        * ! pip install matplotlib
    2. method 2:
        * pip install -m requirements.txt

2. Data location:
    * Modeling_Semantic_Plausibility/Data/pap

