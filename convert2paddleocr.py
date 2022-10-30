import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import glob
import argparse
from tqdm import tqdm

def main(args):
  root_path = glob.glob(f"{args.path}/labels/*")

  train_label = open(f"{args.path}/train_label.txt","w")
  test_label = open(f"{args.path}/test_label.txt","w")
  useen_label = open(f"{args.path}/useen_label.txt","w")
  for file in tqdm(root_path):
      with open(file) as f:
        content = f.readlines()
        f.close()
      content = [x.strip() for x in content]
      text = []
      for i in content:
        label = {}
        i = i.split(',',8)
        label['transcription'] = i[-1]
        label['points'] = [[i[0],i[1]],[i[2],i[3]], [i[4],i[5]],[i[6],i[7]]]
        text.append(label)

      content = []
      text = json.dumps(text, ensure_ascii=False)

      img_name = os.path.basename(file).split('.')[0].split('_')[1]
      int_img = int(img_name)
      img_name = 'im' + "{:04n}".format(int(img_name)) + '.jpg'
      if int_img > 1500:
        useen_label.write( img_name+ '\t'+f'{text}' + '\n')
      elif int_img > 1200:
        test_label.write( img_name+ '\t'+f'{text}' + '\n')
      else:
        train_label.write( img_name+ '\t'+f'{text}' + '\n')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--path', type=str, default="./train_data/vietnamese", 
      help='path to vietnamese dataset')
  args = parser.parse_args()

  main(args)
