import pandas as pd

BASE = "hf://datasets/systemslibrarian/classical-cipher-corpus/"
splits = {'train': 'data/train.jsonl', 'validation': 'data/val.jsonl', 'test': 'data/test.jsonl'}

train = pd.read_json(BASE + splits["train"], lines=True)
valid = pd.read_json(BASE + splits["validation"], lines=True)
test  = pd.read_json(BASE + splits["test"], lines=True)

print(list(train.columns))
print(list(valid.columns))
print(list(test.columns))
