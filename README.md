# BiLSTM-CRF 模型實作

## Steps 1 資料夾介紹

+ `Predict` : 模型Test的輸出
+ `data` : 輸入資料
+ `package` : 模型
+ `saved_model` : 存放模型參數
+ `embedding`：embedding訓練

## Steps 2 虛擬環境建置
+ conda create --name toturial python=3.7
+ pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
+ pip install pandas==1.3.5
+ pip install tqdm==4.62.3
+ pip install genism==4.2.0

## Steps 3 詞嵌入訓練
### Data Format
使用 msra train 作為訓練embedding的資料，字與字中間空一格，範例如下：
```
当 希 望 工 程 救 助 的 百 万 儿 童 成 长 起 来 ， 科 教 兴 国 蔚 然 成 风 时 ， 今 天 有 收 藏 价 值 的 书 你 没 买 ， 明 日 就 叫 你 悔 不 当 初 ！ 
藏 书 本 来 就 是 所 有 传 统 收 藏 门 类 中 的 第 一 大 户 ， 只 是 我 们 结 束 温 饱 的 时 间 太 短 而 已 。 
因 有 关 日 寇 在 京 掠 夺 文 物 详 情 ， 藏 界 较 为 重 视 ， 也 是 我 们 收 藏 北 史 料 中 的 要 件 之 一 。 
```

### Train
```
python word2vec_train.py
```

## Steps 4 BiLSTM-CRF 訓練
### Data Format
使用BIO標記格式，字元與標籤中間空一格，句和句中間空一行，範例如下：
#### Train and Evaluation file
```
淮 B-LOC
科 O
技 O
集 O
市 O
， O
还 O
吸 O
引 O
了 O
联 B-ORG
合 I-ORG
国 I-ORG
工 I-ORG
业 I-ORG
发 I-ORG
展 I-ORG
组 I-ORG
织 I-ORG
中 I-ORG
国 I-ORG
投 I-ORG
资 I-ORG
促 I-ORG
进 I-ORG
处 I-ORG
等 O
国 O
内 O
外 O
十 O
多 O
家 O
知 O
名 O
投 O
资 O
商 O
。 O

他 O
们 O
不 O
仅 O
购 O
买 O
技 O
术 O
， O
而 O
且 O
引 O
进 O
科 O
技 O
人 O
才 O
。 O
```
#### Test file
```
沙
巴
航
空
服
务
中
心

什
么
是
格
鲁
吉
亚
统
一
共
产
党
```
### Parameter

+ `--mode` : Train or Test
+ `--save_model_name` : 存擋 save_model_dir_path 中 模型參數的名稱(.pt格式)
+ `--predict_name` : 存擋 predict 檔案名稱(.txt格式)
+ `--load_model_name` : 讀取的model name
+ `--Train_data_path` : 設定 Training data path
+ `--Eval_data_path` : 設定 Evaluation data path
+ `--Test_data_path` : 設定 Testing data path
+ `--Epoch` : 設定 Training Epoch
+ `--lr` : 設定 learning rate
+ `--batch_size` : 設定 batch size
+ `--lstm_hidden_dim` : 設定 lstm hidden dim
+ `--lstm_dropout_rate` : 設定 lstm dropout rate
+ `--seed` : random seed set
+ `--gpu` : which gpu
+ `--embedding`：詞嵌入檔案位置
+ `--dimension`：詞嵌入維度

## Usage

### Train
```
python main.py --mode Train --save_model_name tutorial.pt  --Epoch 1 --gpu 0
```

### Test
```
python main.py --mode Test --load_model_name tutorial --predict_name predict.txt --gpu 0
```
## Steps 5 Evaluation
先執行turn_to_eval.py，會產生eval.txt，再接續執行conlleval.py，即可得結果score.txt
```
python turn_to_eval.py --truth truth.txt --prediction predict.txt 
python conlleval.py < eval.txt 

```
