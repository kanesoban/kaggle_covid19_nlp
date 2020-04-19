import argparse
import re
import glob
import json
import pandas as pd
from tqdm import tqdm
import nltk
nltk.download('stopwords')


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', default='covid.csv')
    return parser.parse_args()


def main():
    args = parse_args()

    all_json = glob.glob(f'{args.data_path}/**/*.json', recursive=True)

    dict_ = {'paper_id': [], 'full_text': []}

    for idx, entry in tqdm(enumerate(all_json)):
        paper_id, full_text = get_content(entry)
        dict_['paper_id'].append(paper_id)
        dict_['full_text'].append(full_text)

    all_json = glob.glob(f'{args.data_path}/**/*.json', recursive=True)

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

    df_covid.to_csv(args.output_path)


if __name__ == "__main__":
    main()
