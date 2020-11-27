
import pandas as pd
data = pd.read_csv("logvideo_20201013.csv")
import torch

from mmaction2.mmaction.apis import  init_recognizer
from mmcv.parallel import collate, scatter
from mmaction2.mmaction.datasets.pipelines import Compose
import os

import pickle
link2link_encoded = pickle.load( open("link.pkl", "rb"))
fail = pickle.load( open("fail_link.pkl", "rb"))


config = 'mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
# Setup a checkpoint file to load
checkpoint = 'mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

label = 'mmaction2/demo/label_map.txt'
model = init_recognizer(config, checkpoint, device='cuda:0')
cfg = model.cfg
device = next(model.parameters()).device  # model device
# construct label map
with open(label, 'r') as f:
    label = [line.strip() for line in f]
# build the data pipeline
test_pipeline = cfg.data.test.pipeline
test_pipeline = Compose(test_pipeline)
start_index = cfg.data.test.get('start_index', 1)


def encode_video(video):
    vid = dict(
        filename=video,
        label=-1,
        start_index=start_index,
        modality='RGB')

    vid = test_pipeline(vid)
    vid = collate([vid], samples_per_gpu=1)

    vid = scatter(vid, [device])[0]
    with torch.no_grad():
        return model(return_loss=False, **vid)[0]


start_from = 900
for e,link in enumerate(data.link.unique()[start_from:1000]): 
  if link != link:
    fail[e+start_from] = link
    print("fail: "+ str(e+start_from))
    continue

  if (e+start_from)%40 == 0:
    os.system("rm -rf videos")
    print("------- Remove Folder -------")
    os.system("mkdir videos")
    

  print(str(e+start_from) + " ---- " + os.path.basename(os.path.dirname(link)))
  if any([i in  os.path.basename(link) for i in [".mov",".mp4",".MP4",".MOV",".flv",".mkv",".avi",".wmv",".3gp",".webm"]]): 
    cmd = "wget -P videos {}".format(link)
    video_path = "videos/" + os.path.basename(link)
  elif any([i in  os.path.basename(os.path.dirname(link)) for i in [".mov",".mp4",".MP4",".MOV",".flv",".mkv",".avi",".wmv",".3gp",".webm"]]):
    cmd = "wget -P videos {}".format(os.path.dirname(link))
    video_path = "videos/" + os.path.basename(os.path.dirname(link))
  else:
    fail[e+start_from] = link
    print("fail  (not video format): "+ str(e+start_from) + " ---- " + link)
    continue
  os.system(cmd)
  if(not os.path.exists(video_path)):
    fail[e+start_from] = link
    print("fail (forbidden link: "+ str(e+start_from) + " ---- " + link)
    continue
  else:
    link2link_encoded[link] = encode_video(video_path)
  

import pickle
a_file = open("link.pkl", "wb")
pickle.dump(link2link_encoded, a_file)
a_file.close()


fail_file = open("fail_link.pkl", "wb")
pickle.dump(fail, fail_file)
fail_file.close()


import io
out_v = io.open('video_vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('video_meta.tsv', 'w', encoding='utf-8')
for k,v in link2link_encoded.items():
  out_v.write('\t'.join([str(x) for x in v.cpu().numpy()]) + "\n")
  out_m.write(str(k) + "\n")



# [j for j in data.link.unique() if j==j and all([i not in j for i in [".mov",".mp4",".MP4",".MOV",".flv",".jpg",".jpeg",".mkv",".avi",".wmv",".3gp",".webm"]])]