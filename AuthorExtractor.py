import pandas as pd
from os import listdir
from os.path import isfile, join
import json

pd.set_option('display.max_columns', None)


def _finditem(obj, key):
    vals = []
    if key in obj and len(obj[key]) != 0:
        vals.append(obj[key])
    for k, v in obj.items():
        temp = []
        if isinstance(v, dict):
            item = _finditem(v, key)
            if item is not None and len(item) != 0 and item not in vals:
                temp = item
        for s in temp:
            if len(s) != 0 and s not in vals:
                vals.append(str(s))
    return vals


path = 'Resources/Test'

lf = [f for f in listdir(path) if isfile(join(path, f))]
dl = []

for fn in lf:
    with open(join(path, fn)) as json_file:
        data = json.load(json_file)
        # text = " ".join(_finditem(data, 'text'))
        # title = " ".join(_finditem(data, 'title'))
        # desc = " ".join(_finditem(data, 'description'))
        authors = _finditem(data, 'authors')[0] if len(_finditem(data, 'authors')) != 0 else []
        authors += _finditem(data, 'author')
        authors = [a.replace('\n', ' ').lower() for a in authors]
        # dct = {'file': fn.split('_')[1], 'text': text, 'title': title, 'description': desc, 'authors': authors}
        dct = {'file': fn.split('_')[1], 'authors': authors, 'class': None}
        print(dct['file'], dct['authors'])
        dl.append(dct)



df = pd.DataFrame(dl)
# for idx, row in df.iterrows():
#     print(idx, row['file'], row['title'], )
# print(df.shape)

df.to_json("Resources/step2/test/TestAuthors.json", orient="table")
#
# a = pd.read_json("Output/log_tf_1/result_fn@10.json", orient="table")
# b = pd.read_json("Output/tf_idf/result_fn@10.json", orient="table")
#
# print(a)
# print("===============================================")
# print(b)