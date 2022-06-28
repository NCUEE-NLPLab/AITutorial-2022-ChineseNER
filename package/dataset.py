 # -*- coding: utf-8 -*-
import torch
import os,json
from torch.utils.data import Dataset
from package.utils import read_csvdata
import numpy as np

def decode_tags_from_ids(batch_ids,id2tag):
    batch_tags = []
    for ids in batch_ids:
        sequence_tags = []
        for id in ids:
            sequence_tags.append(id2tag[int(id)])
        batch_tags.append(sequence_tags)
    return batch_tags

class Dataset(Dataset):

    def __init__(self,data_path,embedding_path,dimesion,mode='Train'):
        self.mode = mode
        self.data = read_csvdata(data_path,mode=self.mode)
        self.maxlen = 512
        self.embedding = {key: val.values for key, val in embedding_path.T.items()}
        self.dimesion=dimesion
        if  self.mode=='Train':
            self.tag2id = self.auto_get_tag2id()
            self.id2tag = self.auto_get_id2tag(self.tag2id)
            self.save_json()

        elif  self.mode=='Eval' or  self.mode=='Test':
            self.tag2id,self.id2tag = self.load_json()

        print(self.mode)
        print(f"tag2id: {self.tag2id}")
        print(f"id2tag: {self.id2tag}")
        print(f"tag number: {len(self.tag2id)} \n")

    def save_json(self):
        with open('data/tag2id.json','w') as file:
            json.dump(self.tag2id,file)

    def load_json(self):
        with open('data/tag2id.json','r') as file:
            tag2id = json.load(file)
            id2tag = self.auto_get_id2tag(tag2id)
        return tag2id,id2tag

    def auto_get_tag2id(self):
        #設定tag2id
        tag_2_id = dict()
        tagid = 0
        for label_list in self.data['label'].tolist():
            for label in label_list:
                if label not in tag_2_id:
                    tag_2_id[label] = tagid
                    tagid += 1
        return tag_2_id
    
    def auto_get_id2tag(self,tag2id):
        #設定id2tag
        id_2_tag = dict()
        for tag in tag2id:
            id_2_tag[ tag2id[tag] ] = tag
        return id_2_tag

    def collate_fn(self, batch):

        def create_embedding_matrix(word_index,embedding_dict,dimension):
            embedding_matrix=np.zeros((512,dimension))
            for index,word in enumerate(word_index):
                if word in embedding_dict:
                    embedding_matrix[index]=embedding_dict[word]
            return embedding_matrix

        label_list,mask_list=list(),list()
        embedding=torch.tensor(float('nan'))
        
        if self.mode=='Train' or self.mode == 'Eval':
            for char_list,label in batch:
                embedding_matrix=create_embedding_matrix(char_list,embedding_dict=self.embedding,dimension=self.dimesion)
                embedding_matrix = torch.tensor(embedding_matrix)
                if torch.all(torch.isfinite(embedding))== False:
                    embedding=embedding_matrix.view(1,512,self.dimesion)
                else:
                    embedding=torch.cat((embedding, embedding_matrix.view(1,512,self.dimesion)), 0)

                label_spec_list =label 
                label_spec_pad_list = label_spec_list + [0 for _ in range(self.maxlen - len( char_list ) )]
                label_list.append(label_spec_pad_list)

                mask_1 = [1 for _ in range(len(char_list))]
                mask_0 = [0 for _ in range(self.maxlen-( len(char_list)))]
                attention_mask = mask_1 + mask_0
                mask_list.append(attention_mask)

            label_tensor = torch.tensor(label_list)
            mask_tensor = torch.tensor(mask_list,dtype=torch.uint8)
            embedding_tensor=torch.tensor(embedding)

            return embedding_tensor,mask_tensor,label_tensor

        elif self.mode=='Test':
            for char_list in batch:
                embedding_matrix=create_embedding_matrix(char_list,embedding_dict=self.embedding,dimension=300)
                embedding_matrix = torch.tensor(embedding_matrix)
                if torch.all(torch.isfinite(embedding))== False:
                    embedding=embedding_matrix.view(1,512,300)
                else:
                    embedding=torch.cat((embedding, embedding_matrix.view(1,512,300)), 0)

                mask_1 = [1 for _ in range(len(char_list))]
                mask_0 = [0 for _ in range(self.maxlen-( len(char_list)))]
                attention_mask = mask_1 + mask_0
                mask_list.append(attention_mask)

            embedding_tensor = torch.tensor(embedding)
            mask_tensor = torch.tensor(mask_list,dtype=torch.uint8)

            return embedding_tensor,mask_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        sent = item['sentence']

        if self.mode == 'Test': 
            return sent
        else:
            label = [self.tag2id[tag] for tag in item["label"]]
            return sent,label