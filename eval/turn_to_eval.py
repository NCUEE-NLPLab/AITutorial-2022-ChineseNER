#-*- coding=utf-8 -*-
import argparse
import pandas as pd
import codecs
import sys
import csv
from conlleval import *
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
parser = argparse.ArgumentParser()
parser.add_argument("--truth",default='msra_test_truth.txt',type=str,help="your truth")
parser.add_argument("--prediction",default='../Predict/predict.txt',type=str,help='your prediction')
args = parser.parse_args()
args = vars(args)
#read file
df_truth =  pd.read_csv(args["truth"],encoding='utf-8',delimiter=' ',header=None,skip_blank_lines =False, quoting=csv.QUOTE_NONE)
df_prediction =  pd.read_csv(args["prediction"],encoding='utf-8',delimiter=' ',header=None,skip_blank_lines =False, quoting=csv.QUOTE_NONE)
df = pd.concat([df_truth,df_prediction[1]],axis=1)
df.to_csv("eval.txt",encoding='utf-8',header=None,index=None,sep=" ")
