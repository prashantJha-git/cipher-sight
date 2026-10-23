import numpy as np
import pandas as pd
import time
import random
import argparse

import tensorflow as tf
from tensorflow import keras

from pathlib import Path
from rapidfuzz.distance import Levenshtein
from tabulate import tabulate

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" if (BASE_DIR / "data").exists() else BASE_DIR.parent / "data"
RESULTS_DIR = BASE_DIR.parent / "model_results" / "LSTM_results"

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
VOCAB = [PAD, BOS, EOS, " "] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
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
    enc = np.array([encode(s, max_in) for s in frame.ciphertext])
    dec_in = np.array([encode([BOS] + list(s), max_out) for s in frame.plaintext])
    # BOS is not a single character, so build the shifted target separately
    tgt = np.array([encode(list(s) + [EOS], max_out) for s in frame.plaintext])
    return enc, dec_in, tgt


@keras.utils.register_keras_serializable(package="cipher")
def masked_loss(y_true, y_pred):
    """cross-entropy that ignores <pad> positions"""
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = keras.ops.cast(keras.ops.not_equal(y_true, 0), loss.dtype)
    return keras.ops.sum(loss * mask, axis=-1) / keras.ops.maximum(keras.ops.sum(mask, axis=-1), 1.0)


@keras.utils.register_keras_serializable(package="cipher")
class MaskedAccuracy(keras.metrics.Metric):
    """character accuracy that ignores padding and aggregates across batches"""

    def __init__(self, name="masked_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct_tokens = self.add_weight(
            name="correct_tokens", shape=(), initializer="zeros"
        )
        self.total_tokens = self.add_weight(
            name="total_tokens", shape=(), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = keras.ops.cast(y_true, "int32")
        predicted = keras.ops.cast(keras.ops.argmax(y_pred, axis=-1), "int32")
        mask = keras.ops.cast(keras.ops.not_equal(y_true, 0), self.dtype)
        correct = keras.ops.cast(keras.ops.equal(y_true, predicted), self.dtype)

        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, self.dtype)
            if keras.ops.ndim(sample_weight) == 1:
                sample_weight = sample_weight[:, None]
            mask = mask * sample_weight

        self.correct_tokens.assign_add(keras.ops.sum(correct * mask))
        self.total_tokens.assign_add(keras.ops.sum(mask))

    def result(self):
        return keras.ops.divide_no_nan(self.correct_tokens, self.total_tokens)

    def reset_state(self):
        self.correct_tokens.assign(0.0)
        self.total_tokens.assign(0.0)


# LSTM MODEL ->
def build_model(vocab_size, max_in, max_out, emb, hid):
    from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

    enc_in = Input(shape=(max_in,), dtype="int32")
    dec_in = Input(shape=(max_out,), dtype="int32")

    # encoder: reads the ciphertext, hands its final state to the decoder
    enc_emb = Embedding(vocab_size, emb, mask_zero=True)(enc_in)
    _, state_h, state_c = LSTM(hid, return_state=True)(enc_emb)

    # decoder: writes the plaintext one character at a time
    dec_emb = Embedding(vocab_size, emb)(dec_in)
    dec_seq = LSTM(hid, return_sequences=True)(
        dec_emb, initial_state=[state_h, state_c]
    )
    out = Dense(vocab_size, activation="softmax")(dec_seq)
    return keras.Model([enc_in, dec_in], out)

def predict_chunks(model, enc, dec, chunk=512):
    """run model predictions in batches and combine the output arrays"""
    if enc.shape[0] == 0:
        return np.empty((0, dec.shape[1], V), dtype=np.float32)
    if not hasattr(model, "_fast_predict"):
        model._fast_predict = tf.function(
            lambda e, d: model([e, d], training=False), reduce_retracing=True)
    outs = []
    for start in range(0, enc.shape[0], chunk):
        e = tf.constant(enc[start:start + chunk])
        d = tf.constant(dec[start:start + chunk])
        outs.append(model._fast_predict(e, d).numpy())
    return np.concatenate(outs, axis=0)

def decode_batch(model, enc, max_out):
    """greedy decoding many ciphertexts at once"""
    if max_out < 2:
        raise ValueError("max_out must allow at least one character and an EOS token")
    n = enc.shape[0]
    dec = np.zeros((n, max_out), dtype=np.int32)
    dec[:, 0] = STOI[BOS]
    out = np.zeros((n, max_out), dtype=np.int32)
    done = np.zeros(n, dtype=bool)
    for t in range(1, max_out):
        probs = predict_chunks(model, enc, dec)[:, t - 1, :]
        probs[:, STOI[BOS]] = -np.inf
        nxt = probs.argmax(axis=1).astype(np.int32)
        nxt[done] = STOI[PAD]
        out[:, t - 1] = nxt
        done |= nxt == STOI[EOS]
        if done.all():
            break
        dec[:, t] = nxt
    preds = []
    for row in out:
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

    tr_enc, tr_dec, tr_y = make_arrays(train, max_in, max_out)
    va_enc, va_dec, va_y = make_arrays(valid, max_in, max_out)
    te_enc, _, _ = make_arrays(test, max_in, max_out)

    model = build_model(V, max_in, max_out, args.emb, args.hid)
    model.compile(
        optimizer="adam",
        loss=masked_loss,
        metrics=[MaskedAccuracy()],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        CSVLogger(str(result_dir / "history.csv")),
    ]
    start = time.time()
    history = model.fit(
        [tr_enc, tr_dec], tr_y,
        validation_data=([va_enc, va_dec], va_y),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, verbose=2,
    )
    minutes = (time.time() - start) / 60

    # save result/model in model_results/LSTM
    model.save(result_dir / f"{cipher}_LSTM.keras")

    preds = decode_batch(model, te_enc, max_out)
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
    parser.add_argument("--emb", type=int, default=128)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="rows per split, for a quick test run")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.emb < 1 or args.hid < 1:
        parser.error("--epochs, --batch-size, --emb, and --hid must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

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
