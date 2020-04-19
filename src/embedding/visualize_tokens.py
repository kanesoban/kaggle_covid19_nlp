import argparse
import os
import pandas as pd
import numpy as np
import pickle
import plotly.express as px


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--pca_embeddings', required=True)
    parser.add_argument('--tokens', required=True)
    parser.add_argument('--keyword', required=True)
    parser.add_argument('--closest', type=int, required=True)
    parser.add_argument('--output_dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.tokens, 'rb') as f:
        tokens = pickle.load(f)

    with open(args.embeddings, 'rb') as f:
        embeddings = pickle.load(f)

    with open(args.pca_embeddings, 'rb') as f:
        pca_embeddings = pickle.load(f)

    keyword_index = tokens.index(args.keyword)

    X = np.stack([item[1] for item in embeddings.items()], axis=0)
    distances = np.linalg.norm(X - X[keyword_index], axis=1)

    df = pd.DataFrame(pca_embeddings, columns=["Component 1", "Component 2"])
    df["Word"] = tokens
    df["Distance"] = distances

    indices = np.argsort(distances)[:args.closest]

    closest_df = df.iloc[indices]

    with open(os.path.join(args.output_dir, 'tokens_closest_{}.pkl'.format(args.keyword)), 'wb') as f:
        pickle.dump(list(closest_df['Word']), f)

    # Visualize
    #df[df.b.str.contains('^f'), :]
    fig = px.scatter(closest_df, x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="agsunset",size="Distance")
    fig.update_traces(textposition='top center')
    fig.layout.xaxis.autorange = True
    fig.data[0].marker.line.width = 1
    fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
    fig.update_layout(height=800, title_text="2D PCA of bioBERT embeddings", template="plotly_white", paper_bgcolor="#f0f0f0")
    fig.show()


if __name__ == "__main__":
    main()

