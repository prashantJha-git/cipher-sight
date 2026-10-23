import numpy as np
import pandas as pd
import time
import random
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
from rapidfuzz.distance import Levenshtein
from tabulate import tabulate

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" if (BASE_DIR / "data").exists() else BASE_DIR.parent / "data"
RESULTS_DIR = BASE_DIR.parent / "model_results" / "bert_results"

CIPHER_FILES = {
    "caesar": "caesar_gutenberg_50k.csv",
    "columnar": "columnar_transposition_gutenberg_50k.csv",
    "hill": "hill_gutenberg_50k.csv",
    "monoalphabetic": "monoalphabetic_gutenberg_50k.csv",
    "vigenere": "vigenere_gutenberg_50k.csv",
}
# the original sentences (with spaces); needed to restore spaces for columnar/hill
SOURCE_FILE = DATA_DIR / "random_sentences_gutenberg_50k.txt"

# one vocabulary for every cipher
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
CLS, SEP, MASK = "<cls>", "<sep>", "<mask>" # BERT special tokens (added after the shared ids)
VOCAB = [PAD, BOS, EOS, " "] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [CLS, SEP, MASK]
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for c, i in STOI.items()}
V = len(VOCAB)

# DATA LOADING ->
def load_source_sentences():
    lines = [l.strip() for l in SOURCE_FILE.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l]

def build_nospace_lookup(sentences):
    """building a stripped sentence dictionary"""
    index = {}
    for s in sentences:
        index.setdefault(s.replace(" ", ""), set()).add(s)
    return index

def unambiguous_sentences(sentences):
    """keep sentences whose no-space is unique"""
    index = build_nospace_lookup(sentences)
    return sorted({s for hits in index.values() if len(hits) == 1 for s in hits})

def load_cipher_file(cipher):
    return pd.read_csv(DATA_DIR / CIPHER_FILES[cipher], dtype=str, keep_default_na=False)

def sentence_splits(sentences, seed):
    """assign each sentence to test/valid/train once, so every cipher uses the same split"""
    uniq = sorted(set(sentences))
    random.Random(seed).shuffle(uniq)
    n = len(uniq)
    n_test = n_valid = int(0.1 * n)
    test = set(uniq[:n_test])
    valid = set(uniq[n_test:n_test + n_valid])
    return test, valid

# shared helpers
def recover_target(pt, index):
    """restore a unique spaced sentence, ignoring trailing X padding"""
    for k in range(0, 9):   # up to 8 padding letters
        if k and (len(pt) <= k or pt[-k:] != "X" * k):
            break
        hits = index.get(pt[:len(pt) - k])
        if hits:
            return next(iter(hits)) if len(hits) == 1 else None
    return None

def _finish(rows, dropped):
    out = pd.DataFrame(rows, columns=["ciphertext", "plaintext"])
    out.attrs["dropped"] = dropped
    return out

def _spaced_plaintext_preprocessing(df):
    """ciphers whose plaintext already has spaces: just strip the ciphertext."""
    rows = [(ct.replace(" ", ""), pt) for ct, pt in zip(df.ciphertext, df.plaintext)]
    return _finish(rows, 0)

def _unspaced_plaintext_preprocessing(df, index):
    """ciphers whose plaintext lost its spaces / got X padding: restore the real sentence"""
    rows, dropped = [], 0
    for ct, pt in zip(df.ciphertext, df.plaintext):
        target = recover_target(pt, index)
        if target is None:
            dropped += 1
            continue
        rows.append((ct.replace(" ", ""), target))
    return _finish(rows, dropped)

# CAESAR PREPROCESSING
def caesar_preprocessing(df, index):
    return _spaced_plaintext_preprocessing(df)

# COLUMNAR PREPROCESSING
def columnar_preprocessing(df, index):
    return _unspaced_plaintext_preprocessing(df, index)

# HILL PREPROCESSING
def hill_preprocessing(df, index):
    return _unspaced_plaintext_preprocessing(df, index)

# MONOALPHABETIC PREPROCESSING
def monoalphabetic_preprocessing(df, index):
    return _spaced_plaintext_preprocessing(df)

# VIGENERE PREPROCESSING
def vigenere_preprocessing(df, index):
    return _spaced_plaintext_preprocessing(df)

PREPROCESSORS = {
    "caesar": caesar_preprocessing,
    "columnar": columnar_preprocessing,
    "hill": hill_preprocessing,
    "monoalphabetic": monoalphabetic_preprocessing,
    "vigenere": vigenere_preprocessing,
}

# ENCODING / SPLIT ->
def add_split(df, test, valid, allowed):
    """keep only sentences in 'allowed', so every cipher uses the same sentences"""
    df = df[df.plaintext.isin(allowed)]
    split = np.where(df.plaintext.isin(test), "test", np.where(df.plaintext.isin(valid), "valid", "train"))
    return df.assign(split=split)

def encode(text, length):
    ids = [STOI[c] for c in text]
    return ids + [STOI[PAD]] * (length - len(ids))

def make_arrays(frame, max_in, max_out):
    """
    BERT sentence-pair input:  [CLS] ciphertext(max_in) [SEP] [MASK] x max_out
    target: plaintext + EOS for the max_out answer slots (<pad> after that)
    """
    n = len(frame)
    ids = np.full((n, 2 + max_in + max_out), STOI[MASK], dtype=np.int32)
    ids[:, 0] = STOI[CLS]
    ids[:, 1:1 + max_in] = [encode(s, max_in) for s in frame.ciphertext]
    ids[:, 1 + max_in] = STOI[SEP]
    tgt = np.array([encode(list(s) + [EOS], max_out) for s in frame.plaintext])
    return ids, tgt

# BERT MODEL ->
"""
only the maths of BERT is used (no pretrained weights):
   embeddings = LayerNorm(token + position + segment), bidirectional self-attention (NO causal mask,
   only a padding mask), post-LayerNorm blocks, GELU feed-forward, MLM-style prediction head,
   AdamW with weight decay + linear warmup / linear decay
"""
def bert_init():
    return keras.initializers.TruncatedNormal(stddev=0.02)   # BERT's weight init


@keras.utils.register_keras_serializable(package="cipher")
class BertEmbeddings(layers.Layer):
    """LayerNorm(token + position + segment) followed by dropout; segment B starts at seg_start"""

    def __init__(self, vocab_size, max_len, seg_start, d_model, dropout, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size, self.max_len, self.seg_start = vocab_size, max_len, seg_start
        self.d_model, self.dropout = d_model, dropout
        self.tok = layers.Embedding(vocab_size, d_model, embeddings_initializer=bert_init())
        self.pos = layers.Embedding(max_len, d_model, embeddings_initializer=bert_init())
        self.seg = layers.Embedding(2, d_model, embeddings_initializer=bert_init())
        self.norm = layers.LayerNormalization(epsilon=1e-12)
        self.drop = layers.Dropout(dropout)

    def call(self, ids, training=None):
        positions = keras.ops.arange(keras.ops.shape(ids)[1])
        segments = keras.ops.cast(positions >= self.seg_start, "int32")
        x = self.tok(ids) + self.pos(positions) + self.seg(segments)
        return self.drop(self.norm(x), training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(vocab_size=self.vocab_size, max_len=self.max_len, seg_start=self.seg_start,
                   d_model=self.d_model, dropout=self.dropout)
        return cfg


@keras.utils.register_keras_serializable(package="cipher")
class WarmupLinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    """BERT learning-rate schedule: linear warmup to peak_lr, then linear decay to 0"""

    def __init__(self, peak_lr, warmup_steps, total_steps):
        super().__init__()
        self.peak_lr, self.warmup_steps, self.total_steps = peak_lr, warmup_steps, total_steps

    def __call__(self, step):
        step = keras.ops.cast(step, "float32")
        warm = self.peak_lr * step / max(self.warmup_steps, 1)
        decay = self.peak_lr * keras.ops.maximum(self.total_steps - step, 0.0) / max(self.total_steps - self.warmup_steps, 1)
        return keras.ops.where(step < self.warmup_steps, warm, decay)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}


@keras.utils.register_keras_serializable(package="cipher")
class PadAttentionMask(layers.Layer):
    """(batch, q_len, k_len) boolean mask that hides <pad> keys from attention"""

    def __init__(self, q_len, **kwargs):
        super().__init__(**kwargs)
        self.q_len = q_len

    def call(self, ids):
        keep = keras.ops.not_equal(ids, 0)  # (batch, k_len)
        return keras.ops.tile(keep[:, None, :], (1, self.q_len, 1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(q_len=self.q_len)
        return cfg


@keras.utils.register_keras_serializable(package="cipher")
def masked_loss(y_true, y_pred):
    """cross-entropy that ignores <pad> positions"""
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = keras.ops.cast(keras.ops.not_equal(y_true, 0), loss.dtype)
    return keras.ops.sum(loss * mask) / keras.ops.maximum(keras.ops.sum(mask), 1.0)


@keras.utils.register_keras_serializable(package="cipher")
def masked_accuracy(y_true, y_pred):
    """per-character accuracy that ignores <pad> positions"""
    y_true = keras.ops.cast(y_true, "int32")
    hit = keras.ops.cast(keras.ops.equal(y_true, keras.ops.argmax(y_pred, axis=-1)), "float32")
    mask = keras.ops.cast(keras.ops.not_equal(y_true, 0), "float32")
    return keras.ops.sum(hit * mask) / keras.ops.maximum(keras.ops.sum(mask), 1.0)


def bert_block(x, mask, d_model, heads, d_ff, dropout):
    """one BERT encoder layer (post-LayerNorm). attention is fully bidirectional: every position sees every other"""
    a = layers.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, dropout=dropout,
                                  kernel_initializer=bert_init())(x, x, attention_mask=mask)
    a = layers.Dropout(dropout)(a)
    x = layers.LayerNormalization(epsilon=1e-12)(layers.Add()([x, a]))
    f = layers.Dense(d_ff, activation="gelu", kernel_initializer=bert_init())(x)
    f = layers.Dense(d_model, kernel_initializer=bert_init())(f)
    f = layers.Dropout(dropout)(f)
    return layers.LayerNormalization(epsilon=1e-12)(layers.Add()([x, f]))


def build_model(vocab_size, max_in, max_out, args):
    d_model, heads, d_ff = args.d_model, args.heads, args.ff
    n_layers, dropout = args.layers, args.dropout
    seq_len = 2 + max_in + max_out
    answer_start = 2 + max_in               # first [MASK] slot (segment B)

    ids = keras.Input(shape=(seq_len,), dtype="int32")
    mask = PadAttentionMask(seq_len)(ids)   # hides <pad> keys only; no triangular mask

    x = BertEmbeddings(vocab_size, seq_len, answer_start, d_model, dropout)(ids)
    for _ in range(n_layers):
        x = bert_block(x, mask, d_model, heads, d_ff, dropout)

    # prediction head only on the answer slots: dense + GELU + LayerNorm, then vocabulary softmax
    x = layers.Cropping1D(cropping=(answer_start, 0))(x)
    x = layers.Dense(d_model, activation="gelu", kernel_initializer=bert_init())(x)
    x = layers.LayerNormalization(epsilon=1e-12)(x)
    out = layers.Dense(vocab_size, activation="softmax", kernel_initializer=bert_init())(x)
    return keras.Model(ids, out)

def predict_chunks(model, ids, chunk=512):
    """run model predictions in batches and combine the output arrays"""
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if ids.shape[0] == 0:
        return np.empty((0, 0, V), dtype=np.float32)
    if not hasattr(model, "_fast_predict"):
        model._fast_predict = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    outs = []
    for start in range(0, ids.shape[0], chunk):
        outs.append(model._fast_predict(tf.constant(ids[start:start + chunk])).numpy())
    return np.concatenate(outs, axis=0)

def decode_batch(model, ids):
    """all answer slots are filled in one forward pass (no left-to-right loop)"""
    probs = predict_chunks(model, ids)
    for special in (BOS, CLS, SEP, MASK):   # never valid outputs
        probs[:, :, STOI[special]] = -np.inf
    best = probs.argmax(axis=2)
    preds = []
    for row in best:
        chars = []
        for i in row:
            if i in (STOI[PAD], STOI[EOS]):
                break
            chars.append(ITOS[int(i)])
        preds.append("".join(chars))
    return preds

# TRAINING MODEL ON EACH CIPHER ->
def run_cipher(cipher, args, index, all_sentences):
    from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

    # fresh seed + fresh graph, so ciphers never share weights or state
    tf.keras.utils.set_random_seed(args.seed)
    tf.keras.backend.clear_session()

    result_dir = RESULTS_DIR / cipher
    result_dir.mkdir(parents=True, exist_ok=True)

    # data loading + this cipher's own preprocessing
    df = PREPROCESSORS[cipher](load_cipher_file(cipher), index)
    dropped = df.attrs.get("dropped", 0)
    test_set, valid_set = sentence_splits(all_sentences, args.seed)
    df = add_split(df, test_set, valid_set, set(all_sentences))

    if args.limit:  # quick smoke test on a small subset
        df = df.sample(frac=1, random_state=args.seed).groupby("split").head(args.limit)

    train = df[df.split == "train"]
    valid = df[df.split == "valid"]
    test = df[df.split == "test"]
    if train.empty or valid.empty or test.empty:
        raise ValueError(
            f"{cipher} needs non-empty train, validation, and test splits; "
            "check the data or increase --limit"
        )

    max_in = int(df.ciphertext.str.len().max())
    max_out = int(df.plaintext.str.len().max()) + 1 # +1 for EOS

    tr_ids, tr_y = make_arrays(train, max_in, max_out)
    va_ids, va_y = make_arrays(valid, max_in, max_out)
    te_ids, _ = make_arrays(test, max_in, max_out)

    model = build_model(V, max_in, max_out, args)
    # BERT optimizer: AdamW (no decay on biases / LayerNorm) + warmup then linear decay
    total_steps = -(-len(train) // args.batch_size) * args.epochs
    schedule = WarmupLinearDecay(args.lr, int(args.warmup * total_steps), total_steps)
    optimizer = keras.optimizers.AdamW(learning_rate=schedule, weight_decay=args.weight_decay,
                                       beta_1=0.9, beta_2=0.999, epsilon=1e-6, clipnorm=1.0)
    optimizer.exclude_from_weight_decay(var_names=["bias", "gamma", "beta"])
    model.compile(optimizer=optimizer, loss=masked_loss, metrics=[masked_accuracy])

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        CSVLogger(str(result_dir / "history.csv")),
    ]
    start = time.time()
    history = model.fit(
        tr_ids, tr_y,
        validation_data=(va_ids, va_y),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, verbose=2,
    )
    minutes = (time.time() - start) / 60

    # save result/model in model_results/bert_results
    model.save(result_dir / f"{cipher}_bert.keras")

    preds = decode_batch(model, te_ids)
    truths = list(test.plaintext)
    exact = float(np.mean([p == t for p, t in zip(preds, truths)]))
    cer = float(np.mean([Levenshtein.distance(p, t) / max(len(t), 1)
                         for p, t in zip(preds, truths)]))

    pd.DataFrame({"ciphertext": list(test.ciphertext), "true": truths,
                  "pred": preds, "exact": [p == t for p, t in zip(preds, truths)]}
                 ).to_csv(result_dir / "predictions.csv", index=False)

    hist = history.history
    best_epoch = int(np.argmin(hist["val_loss"]))
    return {
        "cipher": cipher,
        "train": len(train), "valid": len(valid), "test": len(test),
        "dropped": dropped,
        "epochs_run": len(hist["loss"]),
        "best_val_acc": round(hist["val_masked_accuracy"][best_epoch], 4),
        "test_exact": round(exact, 4),
        "test_cer": round(cer, 4),
        "minutes": round(minutes, 1),
    }

def ask_user_for_ciphers():
    """interactive menu, used when --ciphers is not given on cmd"""
    names = list(CIPHER_FILES)
    print("which cipher do you want to train on?")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    print("  6. all (one after another, each with its own fresh model)")
    while True:
        raw = input("enter number(s), e.g. 1 or 2,4 : ").replace(",", " ").split()
        try:
            picks = [int(x) for x in raw]
        except ValueError:
            picks = []
        if picks and all(1 <= p <= 6 for p in picks):
            return names if 6 in picks else [names[p - 1] for p in dict.fromkeys(picks)]
        print("invalid choice... try again")

# main ->
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ciphers", nargs="+", choices=list(CIPHER_FILES),
                        help="ciphers to train; if omitted you will be asked")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3,
                        help="early stopping patience")
    parser.add_argument("--d-model", type=int, default=128, help="embedding / hidden size")
    parser.add_argument("--heads", type=int, default=4, help="attention heads (must divide d-model)")
    parser.add_argument("--ff", type=int, default=512, help="feed-forward layer size")
    parser.add_argument("--layers", type=int, default=4, help="BERT encoder blocks")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4, help="peak learning rate")
    parser.add_argument("--warmup", type=float, default=0.1, help="fraction of steps used for lr warmup")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="rows per split, for a quick test run")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size, and --patience must be positive")
    if args.d_model < 1 or args.heads < 1 or args.ff < 1 or args.layers < 1:
        parser.error("--d-model, --heads, --ff, and --layers must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if not 0 <= args.warmup < 1 or args.weight_decay < 0:
        parser.error("--warmup must be in [0, 1) and --weight-decay must be >= 0")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.d_model % args.heads:
        parser.error("--d-model must be divisible by --heads")

    ciphers = args.ciphers or ask_user_for_ciphers()

    missing = [p.name for p in [SOURCE_FILE] + [DATA_DIR / CIPHER_FILES[c] for c in ciphers]
               if not p.exists()]
    if missing:
        raise FileNotFoundError(f"data file(s) not found in data/: {', '.join(missing)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for cipher in CIPHER_FILES:
        (RESULTS_DIR / cipher).mkdir(exist_ok=True)

    all_sentences = unambiguous_sentences(load_source_sentences())
    index = build_nospace_lookup(all_sentences)

    rows = []
    for cipher in ciphers:
        print(f"\n===== {cipher} =====")
        rows.append(run_cipher(cipher, args, index, all_sentences))

    headers = ["cipher", "train", "valid", "test", "dropped", "epochs_run",
               "best_val_acc", "test_exact", "test_cer", "minutes"]
    table = tabulate([[r[h] for h in headers] for r in rows], headers=headers,
                     tablefmt="github")
    print("\n" + table)

    summary_path = RESULTS_DIR / "combined_summary.md"
    summary_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nsaved summary to {summary_path}")

if __name__ == "__main__":
    main()