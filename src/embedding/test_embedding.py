import os
import re
import glob
import json
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn import metrics
from sklearn.utils import shuffle
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

from bert_embeddings import InputExample, bert_embedding_generator


BERT_CONFIG = '/home/cszsolnai/Projects/data/model/bert/biobert_v1.1_pubmed_2/bert_config.json'
VOCAB_FILE = '/home/cszsolnai/Projects/data/model/bert/biobert_v1.1_pubmed_2/vocab.txt'
INIT_CHECKPOINT = '/home/cszsolnai/Projects/data/model/bert/biobert_v1.1_pubmed_2/model.ckpt'
DATA_PATH = '/home/cszsolnai/Projects/data/dataset/CORD-19-research-challenge'
EMBEDDINGS = '/home/cszsolnai/Projects/data/dataset/CORD-19-research-challenge/bioBERT_embeddings'







'''
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = words
df["Distance"] = np.sqrt(df["Component 1"]**2 + df["Component 2"]**2)

indices = np.argsort(distances)[:num]
'''

# Visualize

'''
#df[df.b.str.contains('^f'), :]
'''

'''
fig = px.scatter(df.iloc[indices], x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="agsunset",size="Distance")
fig.update_traces(textposition='top center')
fig.layout.xaxis.autorange = True
fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.update_layout(height=800, title_text="2D PCA of Word2Vec embeddings", template="plotly_white", paper_bgcolor="#f0f0f0")
fig.show()
'''