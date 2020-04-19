import argparse
import pickle
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--articles', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.tokens, 'rb') as f:
        tokens = pickle.load(f)

    df_covid = pd.read_csv(args.articles)
    vectorizer = TfidfVectorizer(vocabulary=tokens, stop_words=stopwords.words('english'), lowercase=True)
    vectors = vectorizer.fit_transform(df_covid['full_text'])

    with open(os.path.join(args.output_dir, 'tf_idf_articles.pkl'), 'wb') as f:
        pickle.dump(vectors, f)


if __name__ == "__main__":
    main()
