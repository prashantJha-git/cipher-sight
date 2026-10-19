import pandas as pd

splits = {'train': 'data/train.jsonl', 'validation': 'data/val.jsonl', 'test': 'data/test.jsonl'}
df = pd.read_json("hf://datasets/systemslibrarian/classical-cipher-corpus/" + splits["train"], lines=True)

print(list(df.columns))
