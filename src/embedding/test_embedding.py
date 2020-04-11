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


def get_content(file_path):
    with open(file_path) as file:
        content = json.load(file)

        # Abstract
        full_text = []
        for entry in content['abstract']:
            full_text.append(entry['text'])
        # Body text
        for entry in content['body_text']:
            full_text.append(entry['text'])

        return content['paper_id'], '\n'.join(full_text)


def lower_case(input_str):
    input_str = input_str.lower()
    return input_str


'''
all_json = glob.glob(f'{DATA_PATH}/**/*.json', recursive=True)

dict_ = {'paper_id': [], 'full_text': []}

for idx, entry in tqdm(enumerate(all_json)):
    paper_id, full_text = get_content(entry)
    dict_['paper_id'].append(paper_id)
    dict_['full_text'].append(full_text)

all_json = glob.glob(f'{DATA_PATH}/**/*.json', recursive=True)

dict_ = {'paper_id': [], 'full_text': []}

for idx, entry in tqdm(enumerate(all_json)):
    paper_id, full_text = get_content(entry)
    dict_['paper_id'].append(paper_id)
    dict_['full_text'].append(full_text)

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'full_text'])
df_covid.drop_duplicates(['paper_id', 'full_text'], inplace=True)

# Remove punctuation

df_covid['full_text'] = df_covid['full_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))

# Convert to lowercase

df_covid['full_text'] = df_covid['full_text'].apply(lambda x: lower_case(x))

df_covid.to_csv('covid.csv')
'''

df_covid = pd.read_csv('covid.csv')

sample = df_covid[:10]


token_embeddings = {}
for converted_article in tqdm(bert_embedding_generator(df_covid['full_text'], list(range(len(df_covid))), BERT_CONFIG, VOCAB_FILE, INIT_CHECKPOINT), desc='Converting article'):
    for token in converted_article:
        if token not in token_embeddings:
            token_embeddings[token] = []

        for embedding in converted_article[token]:
            token_embeddings[token].append(embedding)

        '''
        with open(os.path.join(EMBEDDINGS, token + '.pkl'), 'wb') as f:
            pickle.dump(token_embeddings[token], f)
        '''

with open(os.path.join(EMBEDDINGS, 'token_embeddings.pkl'), 'wb') as f:
    pickle.dump(token_embeddings, f)


'''
with open('token_embeddings.pkl', 'rb') as f:
    token_embeddings = pickle.load(f)

for token in token_embeddings:
    token_embeddings[token] = np.mean(token_embeddings[token], axis=0)

with open('token_embeddings_sample_mean.pkl', 'wb') as f:
    pickle.dump(token_embeddings, f)
'''


'''
with open('token_embeddings_sample_mean.pkl', 'rb') as f:
    token_embeddings = pickle.load(f)
'''


# Filter out non-word tokens
for token in list(token_embeddings.keys()):
    if not re.match(r'[a-zA-Z]+[a-zA-Z0-9]*', token):
        del token_embeddings[token]

# Filter out stopwords
for token in list(token_embeddings.keys()):
    if token in set(stopwords.words('english')):
        del token_embeddings[token]

items = list(token_embeddings.items())

words = [item[0] for item in items]

X = np.stack([item[1] for item in items], axis=0)

# Dimensionality reduction

pca = PCA(n_components=2)
result = pca.fit_transform(X)

with open(os.path.join(EMBEDDINGS, 'pca_result.pkl'), 'wb') as f:
    pickle.dump(result, f)


# Get closest words

# Keyword

keyword = 'illness'

num = 30

distances = np.linalg.norm(X - token_embeddings[keyword], axis=1)

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