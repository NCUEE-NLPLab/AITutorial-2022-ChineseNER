# -*- coding: utf-8 -*-
import json
import pandas as pd
from tqdm import tqdm

def read_csvdata(path,mode='Train'):

    with open(path,'r',encoding='utf-8') as f:
            lines = f.readlines()

    if mode == 'Train' or mode == 'Eval':
  
        data = pd.DataFrame(columns = ['sentence','label'])
        row_dict = dict()
        row_dict['char'],row_dict['label'] = list(),list()
        
        for line in tqdm(lines):
            line = line.strip('\n')
            if line != '' and line != ' ':
                row_dict['char'].append(line.split(' ')[0])
                row_dict['label'].append(line.split(' ')[1])
            else:
                assert len(row_dict['char']) == len(row_dict['label']) , "char 與 label 長度不一"
                data = data.append({'sentence':row_dict['char'],
                                    'label':row_dict['label']},ignore_index=True)
                row_dict['char'],row_dict['label'] = list(),list()

    elif mode == 'Test':

        data = pd.DataFrame(columns = ['sentence'])
        row_dict = dict()
        row_dict['char'] = list()

        for line in tqdm(lines):
            line = line.strip('\n')
            if line != '' and line != ' ':
                row_dict['char'].append(line.split(' ')[0])
            else:
                data = data.append({'sentence':row_dict['char']},ignore_index=True)
                row_dict['char'] = list()
    
    return data


def decode_bio_tags(tags):
    """decode entity (type, start, end) from BIO style tags
    """
    chunks = []
    chunk = [-1, -1, -1]
    for i, tag in enumerate(tags):

        if tag.startswith('B-'):
            if chunk[2] != -1:
                chunks.append(chunk)

            chunk = [-1, -1, -1]
            chunk[0] = tag.split('-')[1]
            chunk[1] = i
            chunk[2] = i + 1
            if i == len(tags) - 1:
                chunks.append(chunk)

        elif tag.startswith('I-') and chunk[1] != -1:
            t = tag.split('-')[1]
            if t == chunk[0]:
                chunk[2] = i + 1

            if i == len(tags) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]

    return chunks
