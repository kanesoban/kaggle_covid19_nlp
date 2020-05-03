import argparse
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfidf_vectorizer', required=True)
    parser.add_argument('--closest_tokens', required=True)
    parser.add_argument('--tf_idf_articles', required=True)
    parser.add_argument('--closest_articles', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.tf_idf_articles, 'rb') as f:
        tf_idf_articles = pickle.load(f)

    with open(args.closest_tokens, 'rb') as f:
        closest_tokens = pickle.load(f)

    with open(args.tfidf_vectorizer, 'rb') as f:
        vectorizer = pickle.load(f)

    df_articles = pd.read_csv(args.articles)

    tf_idf_tokens = vectorizer.transform(closest_tokens)

    n_articles = tf_idf_articles.shape[0]
    n_tokens = len(closest_tokens)

    distances = np.empty((n_tokens, n_articles))
    for i in tqdm(range(n_tokens), desc='Calculating token-article distances'):
        distances[i, :] = np.linalg.norm(tf_idf_articles.toarray() - tf_idf_tokens[i].toarray(), axis=1)

    closest_article_ids = np.argmin(distances, axis=1)

    closest_articles = df_articles.iloc[closest_article_ids]

    closest_articles.to_csv(closest_articles)


if __name__ == "__main__":
    main()
