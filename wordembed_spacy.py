
import pandas as pd
data = pd.read_csv("logvideo_20201013.csv")

import spacy
nlp = spacy.load('vi_spacy_model')
def vectorize_title(doc):
  if type(doc) == float:
    print(doc)
  doc = nlp(doc)
  print(doc)
  return doc.vector

title_ids = data["title"].unique().tolist()
title_ids =  [value for value in title_ids if value == value]
title2title_encoded = {x: vectorize_title(x) for x in title_ids}

