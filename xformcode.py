from transformers import pipeline
import pandas as pd
import os
from mlflow import log_artifact

nlp = pipeline('sentiment-analysis')

def do_nlp(line, arr):
    try:
        global nlp
        s = nlp(line)[0]
    except Exception as err:
        print(f'error occurred while nlp: {err}')
    else:
        arr.append([line, s['label'], s['score']])

def do_nlp_fnx(row):
    global nlp
    try:
        if 'sequence' in row:
            s = nlp(row['sequence'])[0]
        elif 'text' in row:
            s = nlp(row['text'])[0]
        else:
            return '', ''
    except Exception as err:
        print(f'error occurred while nlp: {err}')
    else:
        return [s['label'], s['score']]

# This transform is called for each file in the chosen data
def infin_transform_one_object(filename, output_dir, parentdir, **kwargs):
    print('infin_transform_one_object: Entered. filename=' + filename + ', output_dir=' + output_dir)

    if (filename.endswith('.json')):
        df = pd.read_json(filename, orient='records', typ='frame')
        df[['label', 'score']] = df.apply(do_nlp_fnx, axis=1, result_type='expand')
    else:
        arr = []
        inf = open(filename, 'r', errors='ignore')
        for line in inf.readlines():
            do_nlp(line, arr)
        df = pd.DataFrame(arr, columns=['text', 'label', 'score'])

    print('infin_transform_one_object: shape=' + str(df.shape), flush=True)
    cols = df.columns.values.tolist()
    print('infin_transform_one_object: columns = ' + str(cols), flush=True)
    for index, rw in df.iterrows():
        print('infin_transform_one_object: row = ' + str(rw), flush=True)
    print('infin_transform_one_object: finished creating dataframe', flush=True)
    df.to_json(filename + '.json', orient='records')
    print('infin_transform_one_object: finished writing df to json. file=' + filename + '.json', flush=True)
    log_artifact(filename + '.json', parentdir)
    print('infin_transform_one_object: finished logging artifact', flush=True)
