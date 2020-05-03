import argparse
import re
import os
import glob
import json
import pandas as pd
from tqdm import tqdm
import nltk
import argparse
import re
import numpy as np
import pickle
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from bert_embeddings import bert_embedding_generator


nltk.download('stopwords')


def get_content(file_path):
    with open(file_path) as file:
        content = json.load(file)

        # Abstract
        abstract = []
        for entry in content['abstract']:
            abstract.append(entry['text'])

        full_abstract = '\n'.join(abstract)

        # Body text
        articles_text = []
        for entry in content['body_text']:
            articles_text.append(entry['text'])

        full_text = '\n'.join(articles_text)

        # Authors
        authors = []
        for author in content['metadata']['authors']:
            authors.append(' '.join([author['first'], author['middle'], author['last']]))

        return content['paper_id'], '\n'.join([full_abstract, full_text]), content['metadata'][
            'title'], full_abstract, '\n'.join(authors), full_text


def lower_case(input_str):
    input_str = input_str.lower()
    return input_str


def save_all_articles(data_path, output_dir):
    all_json = glob.glob(f'{data_path}/**/*.json', recursive=True)

    dict_ = {'paper_id': [], 'full_text': []}

    for idx, entry in tqdm(enumerate(all_json)):
        paper_id, concat_text, title, abstract, authors, full_text = get_content(entry)
        dict_['paper_id'].append(paper_id)
        dict_['concat_text'].append(concat_text)
        dict_['title'].append(title)
        dict_['abstract'].append(abstract)
        dict_['authors'].append(authors)
        dict_['full_text'].append(full_text)

    df_covid = pd.DataFrame(dict_, columns=['paper_id', 'concat_text', 'title', 'abstract', 'authors', 'full_text'])
    df_covid.drop_duplicates(['paper_id', 'concat_text', 'title', 'abstract', 'authors', 'full_text'], inplace=True)

    # Remove punctuation
    df_covid['concat_text'] = df_covid['concat_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))

    # Convert to lowercase
    df_covid['concat_text'] = df_covid['concat_text'].apply(lambda x: lower_case(x))

    path = os.path.join(output_dir, 'covid.csv')
    df_covid.to_csv(path)
    return df_covid


def extract_embeddings(df_covid, bert_config, vocab_file, bert_model_checkpoint, embeddings):
    ps = PorterStemmer()

    token_embeddings = {}
    token_counts = {}
    for i, converted_article in tqdm(enumerate(
            bert_embedding_generator(df_covid['concat_text'], list(range(len(df_covid))), bert_config,
                                     vocab_file,
                                     bert_model_checkpoint)), desc='Converting article'):

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

    return token_embeddings


def reduce_dimensionality(embeddings, output_dir, items):
    X = np.stack([item[1] for item in items], axis=0)

    # Dimensionality reduction for displaying
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    return result


def convert_articles_tfidf(tokens, df_covid):
    vectorizer = TfidfVectorizer(vocabulary=tokens, stop_words=stopwords.words('english'), lowercase=True)
    vectors = vectorizer.fit_transform(df_covid['concat_text'])
    return vectors, vectorizer


def find_closest_articles(df_covid, tf_idf_articles, closest_tokens, vectorizer, num_closest_articles):
    tf_idf_tokens = vectorizer.transform(closest_tokens)

    n_articles = tf_idf_articles.shape[0]
    n_tokens = len(closest_tokens)

    distances = np.empty((n_tokens, n_articles))
    for i in tqdm(range(n_tokens), desc='Calculating token-article distances'):
        distances[i, :] = np.linalg.norm(tf_idf_articles.toarray() - tf_idf_tokens[i].toarray(), axis=1)

    closest_article_ids = np.argsort(distances, axis=1)[:num_closest_articles]

    closest_articles = df_covid.iloc[closest_article_ids]

    return closest_articles


def find_closest_tokens_to_keyword(token_embeddings, keyword, num_closest):
    items = list(token_embeddings.items())

    words = [item[0] for item in items]

    X = np.stack([item[1] for item in items], axis=0)

    distances = np.linalg.norm(X - token_embeddings[keyword], axis=1)

    indices = np.argsort(distances)[:num_closest]
    return words[indices]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', default='.')
    parser.add_argument('--bert_config', required=True)
    parser.add_argument('--vocab_file', required=True)
    parser.add_argument('--bert_model_checkpoint', required=True)
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--keyword', required=True)
    parser.add_argument('--num_closest_keywords', default=30, type=int)
    parser.add_argument('--num_closest_articles', default=10, type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    df_covid = save_all_articles(args.data_path, args.output_dir)

    token_embeddings = extract_embeddings(df_covid, args.bert_config, args.vocab_file, args.bert_model_checkpoint, args.embeddings)

    with open(args.embeddings, 'wb') as f:
        pickle.dump(token_embeddings, f)

    items = list(token_embeddings.items())

    words = [item[0] for item in items]
    with open(os.path.join(args.output_dir, 'tokens.pkl'), 'wb') as f:
        pickle.dump(words, f)

    pca_token_embeddings = reduce_dimensionality(token_embeddings, args.output_dir, items)

    with open(os.path.join(args.output_dir, 'pca_token_embeddings_{}.pkl'.format(2)), 'wb') as f:
        pickle.dump(pca_token_embeddings, f)

    '''
    with open(args.embeddings, 'rb') as f:
        token_embeddings = pickle.load(f)
    '''

    with open(args.tokens, 'rb') as f:
        tokens = pickle.load(f)

    vectors, vectorizer = convert_articles_tfidf(tokens, df_covid)

    with open(os.path.join(args.output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join(args.output_dir, 'tf_idf_articles.pkl'), 'wb') as f:
        pickle.dump(vectors, f)

    with open(args.tf_idf_articles, 'rb') as f:
        tf_idf_articles = pickle.load(f)

    '''
    with open(args.closest_tokens, 'rb') as f:
        closest_tokens = pickle.load(f)
    '''

    '''
    with open(args.tfidf_vectorizer, 'rb') as f:
        vectorizer = pickle.load(f)
    '''

    closest_tokens = find_closest_tokens_to_keyword(token_embeddings, args.keyword, args.num_closest)

    closest_articles = find_closest_articles(df_covid, tf_idf_articles, closest_tokens, vectorizer, args.num_closest_articles)


if __name__ == "__main__":
    main()
