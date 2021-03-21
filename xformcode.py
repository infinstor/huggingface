from transformers import pipeline
import pandas as pd
import os
from mlflow import log_artifact

nlp = pipeline('sentiment-analysis')

# This transform is called for each file in the chosen data
def infin_transform_one_object(filename, output_dir, parentdir, **kwargs):
    print('infin_transform_one_object: Entered. filename=' + filename + ', output_dir=' + output_dir)
    global nlp

    df = pd.DataFrame(columns=['text', 'sentiment_label', 'sentiment_score'], )

    inf = open(filename, 'r', errors='ignore')
    for line in inf.readlines():
        s = nlp(line)[0]
        df.append({'text': line, 'sentiment_label': s['label'], 'sentiment_score': s['score']}, ignore_index=True)

    df.to_json(filename + '.json')
    log_artifact(filename + '.json', parentdir)
