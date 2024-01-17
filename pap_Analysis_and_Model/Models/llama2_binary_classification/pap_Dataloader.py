from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import torch

class pap_Dataloader(object):
    '''
    Process the pap data file and return a dataloader containing tokenized text and 0,1 labels.
    @author: Li Lin.
    '''
    def __init__(self,filepath_name,tokenizer,batch_size='defalut'):
        '''
        Initializes the pap data file Dataloader with the dataset name(file path), tokenizer, and optional batch size.
        : Args:
            failpath_name (str): The name of the data file path.
            tokenizer (Tokenizer): The tokenizer to be used for tokenizing the text.
            batch_size (int, optional): The batch size for the dataloader. Its defalut the max length of the dataset.
        '''
        self.filepath_name=filepath_name
        self.dataset=self.get_dataset()
        self.new_dataset=self.rebuilt_dataset()
        self.tokenized_dataset=self.tokenize_text(tokenizer)
        if batch_size=='defalut':
            self.batch_size=len(self.dataset)
        else:
            self.batch_size=batch_size
        self.dataloader=self.get_dataloader()


    def get_dataset(self):
        '''
        Loads the dataset from the CSV file.
        : Returns: The loaded dataset.
        '''
        return load_dataset("csv", data_files = self.filepath_name, split = "train")
    

    def rebuilt_dataset(self):
        '''
        Rebuilds the dataset by mapping labels to numerical values.
        : Returns: new_dataset: The rebuilt dataset.
        '''
        label2id = {"implausible": 0, "plausible": 1}
        dataset_labels = [label2id[label] for label in self.dataset["original_label"]]

        new_dataset =  Dataset.from_dict({
        'text': self.dataset['text'],
        'labels': dataset_labels # Update 'original_label'
        })
        print(new_dataset)
        return new_dataset #,dataset_labels
    

    def tokenize_text(self, tokenizer):
        '''
        Tokenizes the text in the dataset using the provided tokenizer from llama.
        : Args: tokenizer (Tokenizer): The tokenizer to be used.
        : Returns: tokenized_datasets: The tokenized dataset.
        '''
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding=True, truncation=True,return_tensors="pt").to("cuda") 
        tokenized_datasets= self.new_dataset.map(tokenize_function, batched=True,batch_size=len(self.new_dataset))
        tokenized_datasets  =tokenized_datasets.remove_columns(["text"])
        return tokenized_datasets

    def get_dataloader(self):
        '''
        Creates and returns a dataloader for the tokenized dataset.
        : Returns: dataLoader
        '''
        def collate(batch):  #to padding the features data
            labels=[]
            ids_all=[]
            attention_mask=[]
            for i in batch:
                a=torch.tensor(i['input_ids'])
                b=torch.tensor(i['attention_mask'])
                c=torch.tensor(i['labels'])
                labels.append(c) #(label)
                ids_all.append(a) #(features)
                attention_mask.append(b) #(torch.LongTensor(len(features)))

            batch_of_labels=torch.stack(labels)
            batch_of_attention_mask=torch.stack(attention_mask)
            batch_of_ids=torch.stack(ids_all)
            batch_of_text={'input_ids':batch_of_ids,'attention_mask':batch_of_attention_mask}

            return batch_of_labels,batch_of_text

        dataloader = DataLoader(self.tokenized_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=collate)
        return dataloader
        