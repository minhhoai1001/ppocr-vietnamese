import json
import os
import cv2
import copy
import numpy as np
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image
import argparse
from tqdm import tqdm

def print_draw_crop_rec_res( img_crop_list, img_name, crop_label, text):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
          crop_name=img_name+'_'+str(bno)+'.jpg'
          crop_name_w = "./train_data/vietnamese/img_crop/{}".format(crop_name)
          cv2.imwrite(crop_name_w, img_crop_list[bno])
          crop_label.write("{0}\t{1}\n".format(crop_name, text[bno]))

def main(args):
  crop_label = open(f'{args.path}/crop_label.txt','w')
  with open(f'{args.path}/train_label.txt','r') as file_text:
    img_files=file_text.readlines()
    
  count=0
  for img_file in tqdm(img_files):
    content = json.loads(img_file.split('\t')[1].strip())

    dt_boxes=[]
    text=[]
    
    for i in content:
      content = i['points']
      if i['transcription'] == "###":
        count+=1
        continue
      bb = np.array(i['points'],dtype=np.float32)
      dt_boxes.append(bb)
      text.append(i['transcription'])

    image_file = f'{args.path}/train_images/' +img_file.split('\t')[0]
    img = cv2.imread(image_file)
    ori_im=img.copy()
    img_crop_list=[]

    for bno in range(len(dt_boxes)):
      tmp_box = copy.deepcopy(dt_boxes[bno])
      img_crop = get_rotate_crop_image(ori_im, tmp_box)
      img_crop_list.append(img_crop)
    img_name = img_file.split('\t')[0].split('.')[0]
    
    if not os.path.exists(f'{args.path}/img_crop'):
      os.mkdir(f'{args.path}/img_crop')
    print_draw_crop_rec_res(img_crop_list, img_name, crop_label, text)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--path', type=str, default="./train_data/vietnamese", 
      help='path to vietnamese dataset')
  args = parser.parse_args()

  main(args)