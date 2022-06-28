import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import os,random,argparse
from package.model import BiLSTM_CRF
from package.dataset import Dataset, decode_tags_from_ids
from eval.conlleval import evaluate
import pandas as pd

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()

#使用模式
parser.add_argument("--mode",default="Train",type=str,help="Train or Test")
#預存檔在 save_model_dir_path 中 模型參數的名稱
parser.add_argument("--save_model_name",default="tutorial.pt",type=str,help="存擋 save_model_dir_path 中 模型參數的名稱(.pt格式)")
#predict file name
parser.add_argument("--predict_name",default="predict.txt",type=str,help="存擋 predict 檔案名稱(.txt格式)")
#讀取 saved_model 裡模型參數的完整路徑
parser.add_argument("--load_model_name",default="tutorial",type=str,help="讀取的model name")
#設定 Training data path
parser.add_argument("--Train_data_path",default="data/msra_train.txt",type=str,help="設定 Training data path")
#設定 Evaluation data path
parser.add_argument("--Eval_data_path",default="data/msra_eval.txt",type=str,help="設定 Evaluation data path")
#設定 Testing data path
parser.add_argument("--Test_data_path",default="data/msra_test.txt",type=str,help="設定 Testing data path")
#設定 Training Epoch
parser.add_argument("--Epoch",default=40, type=int,help="設定 Training Epoch")
#設定 learning rate
parser.add_argument("--lr",default=4e-3,type=float,help="設定 learning rate")
#設定 batch size
parser.add_argument("--batch_size",default=256,type=int,help="設定 batch size")
#設定 lstm hidden dim
parser.add_argument("--lstm_hidden_dim",default=1024,type=int,help="設定 lstm hidden dim")
#設定 lstm dropout rate
parser.add_argument("--lstm_dropout_rate",default=0.1, type=int,help="設定 lstm dropout rate")
# 設定Random seed
parser.add_argument("--seed",default=87,type=int,help="random seed set")
#選擇gpu
parser.add_argument("--gpu",default='0',type=str,help="which gpu")
#pretrai nembedding
parser.add_argument("--embedding",default='embedding/msra_word2vec.txt',type=str,help="which gpu")
#embedding dimesion
parser.add_argument("--dimesion",default=300,type=int,help="embedding dimension")
args = parser.parse_args()
args = vars(args)

#設定gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']  # 指定GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 設定隨機種子值，以確保輸出是確定的
seed_val = args['seed']
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Train or Test or Predict
mode = args['mode']

#hyper parameter
EPOCH = args['Epoch']
lr = args['lr']
batch_size = args['batch_size']
lstm_hidden_dim = args['lstm_hidden_dim']
lstm_dropout_rate = args['lstm_dropout_rate']
dimesion=args['dimesion']
roberta_freeze = True
word2vec = pd.read_csv(args['embedding'], sep=" ", quoting=3, header=None, index_col=0,skiprows=1)
word2vec_embedding = {key: val.values for key, val in word2vec.T.items()}

#Training Mode
if mode == 'Train':
     
    #create dir for save model
    assert len(args['save_model_name'].split('.pt')) == 2 ,'模型名稱錯誤:{xxx.pt}'
    dir_name = args['save_model_name'].split('.pt')[0]

    model_path = os.path.join('saved_model',dir_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    #save parameter
    with open(os.path.join(model_path,'parameter.txt'),'w',encoding='utf-8') as p_file:
        for item in args:
            p_file.write(f"{item}: {args[item]}\n")

    #Load Training data
    print("Loading Training data")
    Train_data_path = args['Train_data_path']
    dataset_train = Dataset(Train_data_path,word2vec,dimesion,mode='Train')
    dataloader_train = DataLoader(dataset_train, collate_fn=dataset_train.collate_fn,
                                batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"training data size : {dataset_train.__len__()}\n")

    #Load Eval Data
    print("Loading Evaluation data")
    Eval_data_path = args['Eval_data_path']
    dataset_eval = Dataset(Eval_data_path,word2vec,dimesion,mode='Eval')
    dataloader_eval = DataLoader(dataset_eval, collate_fn=dataset_eval.collate_fn,
                                batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"evaluation data size : {dataset_eval.__len__()} \n")

    #Create Model
    model = BiLSTM_CRF(len(dataset_train.id2tag),lstm_hidden_dim=lstm_hidden_dim,lstm_dropout_rate=lstm_dropout_rate).to(device)
    model.reset_parameters()

    #Create optimizer
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    for epoch in range(EPOCH):
        print(f"\nEpoch: {epoch}=====================================================================================")

        #Training
        model.train()
        print("---------------------------Training---------------------------")
        with tqdm(desc='Train', total=len(dataloader_train)) as t:
            for i, (input,mask,label) in enumerate(dataloader_train):
                input,mask, label = [_.to(device) for _ in (input,mask,label)]
                loss = model(input,mask,label)
                loss.backward()

                #更新tqdm資訊
                t.update(1)
                t.set_postfix(loss=float(loss))

                # 梯度裁剪，避免出現梯度爆炸情況
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
        
        #Eval
        model.eval()
        print('--------------------------Evaluation--------------------------')
        pred_list = list()
        true_list = list()
        with torch.no_grad():

            for i, (input,mask,label) in enumerate(tqdm(dataloader_eval, desc='Eval')):
                #丟入模型預測
                input, mask, label = [_.to(device) for _ in (input,mask,label)]
                y_pred = model.loss(input,mask)
                
                #抓模型預測label
                y_pred = decode_tags_from_ids(y_pred,dataset_eval.id2tag)
                for sent in y_pred:
                    for tag in sent:
                        pred_list.append(tag)

                #將pad的label消除
                y_true = list()
                for s in range(len(y_pred)):
                    sentence_list = list()
                    for p,l in zip(y_pred[s],label[s]):
                        sentence_list.append(l)
                    y_true.append(sentence_list)

                #抓答案
                y_true = decode_tags_from_ids(y_true,dataset_eval.id2tag)
                for sent in y_true:
                    for tag in sent:
                        true_list.append(tag)
            
            #印出評分
            prec, rec, f1 = evaluate(true_list,pred_list,file_name=model_path)
            print(f"f1: {f1}")
            print(f"recall: {rec}")
            print(f"precision: {prec}")

    #儲存model
    model_arg_file_name = args['save_model_name']
    save_model_name = os.path.join(model_path,model_arg_file_name)
    torch.save(model.state_dict(), save_model_name)
    print('Training Finish')

#Predict Mode
elif mode == 'Test':

    #load model name
    load_model_path = os.path.join('saved_model',args['load_model_name'],args['load_model_name']+'.pt')

    #pred txt name
    Pred_save_path = os.path.join( 'Predict' ,args['predict_name'])

    #Load Test Data
    print("Loading Testing data")
    Test_data_path = args['Test_data_path']
    dataset_test = Dataset(Test_data_path,word2vec,dimesion,mode='Test')
    dataloader_test = DataLoader(dataset_test, collate_fn=dataset_test.collate_fn,
                                batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"testing data size : {dataset_test.__len__()} \n")

    #Load PreTrain Model
    model = BiLSTM_CRF(len(dataset_test.id2tag),lstm_hidden_dim=lstm_hidden_dim, lstm_dropout_rate=lstm_dropout_rate).to(device)
    model.load_state_dict(torch.load(load_model_path))      
    #Test
    model.eval()
    print('--------------------------Testing--------------------------')

    with torch.no_grad():
        sum_pred_list = list()

        for i, (input, mask) in enumerate(tqdm(dataloader_test, desc='Test')):
            input, mask = [_.to(device) for _ in (input, mask)]
            y_pred = model.loss(input, mask)
            y_pred = decode_tags_from_ids(y_pred,dataset_test.id2tag)
            sum_pred_list += y_pred


        with open(Pred_save_path,'w',encoding='utf-8') as file:
            for char_list,pred_list in zip(dataset_test.data['sentence'],sum_pred_list):
                for char,pred in zip(char_list,pred_list):
                    row = f"{char} {pred}\n"
                    file.write(row)
                file.write('\n')
                        

