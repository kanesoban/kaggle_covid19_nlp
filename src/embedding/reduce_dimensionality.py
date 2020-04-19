import argparse
import os
import numpy as np
import pickle
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dimensions', default=2, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.embeddings, 'rb') as f:
        token_embeddings = pickle.load(f)

    items = list(token_embeddings.items())
    
    words = [item[0] for item in items]
    with open(os.path.join(args.output_dir, 'tokens.pkl'), 'wb') as f:
        pickle.dump(words, f)
    
    X = np.stack([item[1] for item in items], axis=0)
    
    # Dimensionality reduction
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    
    with open(os.path.join(args.output_dir, 'pca_token_embeddings_{}.pkl'.format(args.dimensions)), 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
