from transformers import pipeline
import pandas as pd
import os
from mlflow import log_artifact

nlp = pipeline('sentiment-analysis')

# This transform is called for each file in the chosen data
def infin_transform_one_object(filename, output_dir, parentdir, **kwargs):
    print('infin_transform_one_object: Entered. filename=' + filename + ', output_dir=' + output_dir)

    arr = []
    inf = open(filename, 'r', errors='ignore')
    for line in inf.readlines():
        one = []
        try:
            global nlp
            s = nlp(line)[0]
        except Exception as err:
            print(f'error occurred while nlp: {err}')
        else:
            arr.append([line, s['label'], s['score']])

    df = pd.DataFrame(arr, columns=['text', 'sentiment_label', 'sentiment_score'])
    print('infin_transform_one_object: finished creating dataframe', flush=True)
    df.to_json(filename + '.json', orient='records')
    print('infin_transform_one_object: finished writing to df to json. file=' + filename + '.json', flush=True)
    log_artifact(filename + '.json', parentdir)
    print('infin_transform_one_object: finished logging artifact', flush=True)
