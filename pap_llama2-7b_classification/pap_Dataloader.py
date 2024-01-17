from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import torch

class pap_Dataloader(object):
    def __init__(self,dataset_name,tokenizer,batch_size='defalut'):
        self.dataset_name=dataset_name
        self.dataset=self.get_dataset()
        self.new_dataset=self.rebuilt_dataset()
        self.tokenized_dataset=self.tokenize_text(tokenizer)
        if batch_size=='defalut':
            self.batch_size=len(self.dataset)
        else:
            self.batch_size=batch_size
        self.dataloader=self.get_dataloader()


    def get_dataset(self):
        return load_dataset("csv", data_files = self.dataset_name , split = "train")
    

    def rebuilt_dataset(self):
        label2id = {"implausible": 0, "plausible": 1}
        dataset_labels = [label2id[label] for label in self.dataset["original_label"]]

        new_dataset =  Dataset.from_dict({
        'text': self.dataset['text'],
        'labels': dataset_labels # Update 'original_label'
        })
        print(new_dataset)
        return new_dataset #,dataset_labels
    

    def tokenize_text(self, tokenizer):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding=True, truncation=True,return_tensors="pt").to("cuda") 
        print(self.new_dataset)
        tokenized_datasets= self.new_dataset.map(tokenize_function, batched=True,batch_size=len(self.new_dataset))
        tokenized_datasets  =tokenized_datasets.remove_columns(["text"])
        return tokenized_datasets
    

    def print_tokenized_datasets_info(self,tokenized_datasets): #for debug
        print(self.tokenized_datasets [0])
        print(type(tokenized_datasets [0]))
        print(tokenized_datasets [0]['input_ids'])   
        print(type(tokenized_datasets [0]['input_ids']))
        print(tokenized_datasets[0]['input_ids'][0])
        print(type(tokenized_datasets[0]['input_ids'][0]))


    def get_dataloader(self):
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
        