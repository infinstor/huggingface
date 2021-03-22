from transformers import pipeline
import pandas as pd
import os
from mlflow import log_artifact

nlp = pipeline('sentiment-analysis')

# This transform is called for each file in the chosen data
def infin_transform_one_object(filename, output_dir, parentdir, **kwargs):
    print('infin_transform_one_object: Entered. filename=' + filename + ', output_dir=' + output_dir)
    global nlp

    df = pd.DataFrame(columns=['text', 'sentiment_label', 'sentiment_score'])
    print('infin_transform_one_object: finished creating empty dataframe', flush=True)

    inf = open(filename, 'r', errors='ignore')
    print('infin_transform_one_object: finished opening input file for read', flush=True)
    for line in inf.readlines():
        print('infin_transform_one_object: read line=' + line, flush=True)
        s = nlp(line)[0]
        print('infin_transform_one_object: finished nlp on line. result=' + str(s), flush=True)
        df.append({'text': line, 'sentiment_label': s['label'], 'sentiment_score': s['score']}, ignore_index=True)
        print('infin_transform_one_object: added line', flush=True)

    df.to_json(filename + '.json')
    print('infin_transform_one_object: finished writing to json. file=' + filename + '.json', flush=True)
    log_artifact(filename + '.json', parentdir)
    print('infin_transform_one_object: finished logging artifact', flush=True)
