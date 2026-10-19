import pandas as pd

splits = {'train': 'data/train.jsonl', 'validation': 'data/val.jsonl', 'test': 'data/test.jsonl'}
BASE = "hf://datasets/systemslibrarian/classical-cipher-corpus/"

train = pd.read_json(BASE + splits["train"], lines=True)
valid = pd.read_json(BASE + splits["validation"], lines=True)
test  = pd.read_json(BASE + splits["test"], lines=True)

print(list(train.columns))
