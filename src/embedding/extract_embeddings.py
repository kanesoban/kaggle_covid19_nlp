import argparse
import re
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')


from bert_embeddings import bert_embedding_generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--articles_csv', required=True)
    parser.add_argument('--bert_config', required=True)
    parser.add_argument('--vocab_file', required=True)
    parser.add_argument('--bert_model_checkpoint', required=True)
    parser.add_argument('--embeddings', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    df_covid = pd.read_csv('covid.csv')

    ps = PorterStemmer()

    token_embeddings = {}
    token_counts = {}
    for i, converted_article in tqdm(enumerate(
            bert_embedding_generator(df_covid['concat_text'], list(range(len(df_covid))), args.bert_config,
                                     args.vocab_file,
                                     args.bert_model_checkpoint)), desc='Converting article'):
        paper_id = df_covid['paper_id'].iloc[i]

        for token in converted_article:
            if not re.match(r'[a-zA-Z]+[a-zA-Z0-9]*', token):
                continue

            # Filter out 1-2 long tokens
            if len(token) <= 2:
                continue

            # Filter out stopwords
            if token in set(stopwords.words('english')):
                continue

            token = ps.stem(token)

            embedding = np.sum(converted_article[token], axis=0)
            count = len(converted_article[token])
            if token in token_embeddings:
                token_embeddings[token] += embedding
                token_counts[token] += count
            else:
                token_embeddings[token] = embedding
                token_counts[token] = count

    for token in token_embeddings:
        token_embeddings[token] /= token_counts[token]

    with open(args.embeddings, 'wb') as f:
        pickle.dump(token_embeddings, f)


if __name__ == "__main__":
    main()
