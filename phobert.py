
import torch
import re
from vncorenlp import VnCoreNLP
from transformers import AutoModel, AutoTokenizer


import pandas as pd
data = pd.read_csv("logvideo_20201013.csv")

rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
phobert = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def vectorize_title(sentence):
  # To perform word (and sentence) segmentation

  sentence = rdrsegmenter.tokenize(re.sub(r'[^àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝa-zA-Z0-9\s]', '', str(sentence), flags=re.MULTILINE)+"."
)
  #print(len(sentence[0]),"sss")
  sentence = " ".join(sentence[0][:50])
  #if(len(sentence) > 1):
  
  input_ids = torch.tensor([tokenizer.encode(sentence,max_length=256,truncation=True)])
  with torch.no_grad():
    return phobert(input_ids)[1].numpy()[0]  # Models outputs are now tuples


title_ids = data["title"].unique().tolist()
title_ids =  [value for value in title_ids if value == value and len(value) >0]
title2title_encoded = {x: vectorize_title(x) for x in title_ids}
import pickle
a_file = open("title.pkl", "wb")
pickle.dump(title2title_encoded, a_file)


