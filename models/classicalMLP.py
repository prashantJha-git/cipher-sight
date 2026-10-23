import numpy as np
import pandas as pd
import time
import random
import argparse

from pathlib import Path
from rapidfuzz.distance import Levenshtein
from tabulate import tabulate

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" if (BASE_DIR / "data").exists() else BASE_DIR.parent / "data"
RESULTS_DIR = BASE_DIR.parent / "model_results" / "classicalMLP_results"

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
    for k in range(9):
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
    """ciphers whose plaintext already has spaces: just strip the ciphertext"""
    rows = [(ct.replace(" ", ""), pt) for ct, pt in zip(df.ciphertext, df.plaintext)]
    return _finish(rows, 0)


def _unspaced_plaintext_preprocessing(df, index):
    """ciphers whose plaintext lost spaces / got X padding: restore the real sentence"""
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
    split = np.where(
        df.plaintext.isin(test),
        "test",
        np.where(df.plaintext.isin(valid), "valid", "train"),
    )
    return df.assign(split=split)


def one_hot_encode(texts, seq_len):
    """encode strings as flattened one-hot character sequences"""
    too_long = [text for text in texts if len(text) > seq_len]
    if too_long:
        raise ValueError(
            f"sequence length {seq_len} is too short for text of length {len(too_long[0])}"
        )

    arr = np.zeros((len(texts), seq_len, V), dtype=np.uint8)
    for i, text in enumerate(texts):
        for j, char in enumerate(text):
            try:
                arr[i, j, STOI[char]] = 1.0
            except KeyError as exc:
                raise ValueError(f"unsupported character {char!r} in input text") from exc
        for j in range(len(text), seq_len):
            arr[i, j, STOI[PAD]] = 1.0
    return arr.reshape(len(texts), -1)


def encode_targets(texts, seq_len):
    targets = one_hot_encode(texts, seq_len).reshape(len(texts), seq_len, V)
    for i, text in enumerate(texts):
        if len(text) >= seq_len:
            raise ValueError(
                f"sequence length {seq_len} must leave room for EOS after"
                f"text of length {len(text)}"
            )
        targets[i, len(text), STOI[PAD]] = 0.0
        targets[i, len(text), STOI[EOS]] = 1.0
    return targets.reshape(len(texts), -1)


# CLASSICAL MLP ->
class ClassicalMLP:
    def __init__(self, input_size, hidden_size, output_size, seq_len, vocab_size,
                 reward_points=1.0, penalty_points=1.0, momentum=0.9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.reward_points = reward_points
        self.penalty_points = penalty_points
        self.total_reward = 0.0
        self.total_penalty = 0.0

        # he initialization for Leaky ReLU
        self.weights_input_hidden = np.random.randn(
            self.input_size, self.hidden_size
        ).astype(np.float32) * np.sqrt(2.0 / self.input_size)
        self.bias_hidden = np.zeros((1, self.hidden_size), dtype=np.float32)

        self.weights_hidden_output = np.random.randn(
            self.hidden_size, self.output_size
        ).astype(np.float32) * np.sqrt(2.0 / self.hidden_size)
        self.bias_output = np.zeros((1, self.output_size), dtype=np.float32)

        self.momentum = momentum
        self.v_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.v_bias_hidden = np.zeros_like(self.bias_hidden)
        self.v_weights_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.v_bias_output = np.zeros_like(self.bias_output)

    def leaky_relu(self, x):
        return np.maximum(0.01 * x, x)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1.0, 0.01)

    def softmax_per_char(self, x):
        """softmax over vocabulary dimension for each character position"""
        n = x.shape[0]
        x3d = x.reshape(n, self.seq_len, self.vocab_size)
        x_max = np.max(x3d, axis=2, keepdims=True)
        exp_x = np.exp(x3d - x_max)
        probs = exp_x / (np.sum(exp_x, axis=2, keepdims=True) + 1e-9)
        return probs.reshape(n, -1)

    def feed_forward(self, x):
        self.hidden_activation = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.leaky_relu(self.hidden_activation)

        self.output_activation = (
            np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        )
        self.predicted_output = self.softmax_per_char(self.output_activation)
        return self.predicted_output

    def reward_penalty(self, predicted_output, y):
        """
        compare argmax character against true character:
          correct char -> +reward_points
          wrong char   -> -penalty_points
        """
        n = predicted_output.shape[0]
        pred_idx = predicted_output.reshape(n, self.seq_len, self.vocab_size).argmax(axis=2)
        true_idx = y.reshape(n, self.seq_len, self.vocab_size).argmax(axis=2)

        active = true_idx != STOI[PAD]
        correct = (pred_idx == true_idx) & active
        points = np.where(
            active,
            np.where(correct, self.reward_points, -self.penalty_points),
            0.0,
        )
        return correct, points

    def backward(self, X, y, learning_rate):
        n = X.shape[0]

        correct, points = self.reward_penalty(self.predicted_output, y)
        self.total_reward += float(points[points > 0].sum())
        self.total_penalty += float(-points[points < 0].sum())

        active = y.reshape(n, self.seq_len, self.vocab_size).argmax(axis=2) != STOI[PAD]
        token_count = np.maximum(active.sum(axis=1, keepdims=True), 1)
        multiplier = np.where(
            correct,
            1.0 / (1.0 + self.reward_points),
            1.0 + self.penalty_points,
        )
        multiplier = multiplier * active / token_count
        multiplier = np.repeat(multiplier, self.vocab_size, axis=1)

        output_error = (y - self.predicted_output) * multiplier

        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.leaky_relu_derivative(self.hidden_activation)

        scale = 1.0 / n
        grad_w_out = np.dot(self.hidden_output.T, output_error) * scale
        grad_b_out = np.sum(output_error, axis=0, keepdims=True) * scale
        grad_w_in = np.dot(X.T, hidden_delta) * scale
        grad_b_in = np.sum(hidden_delta, axis=0, keepdims=True) * scale

        beta = self.momentum
        self.v_weights_hidden_output = (
            beta * self.v_weights_hidden_output + (1.0 - beta) * grad_w_out
        )
        self.v_bias_output = beta * self.v_bias_output + (1.0 - beta) * grad_b_out
        self.v_weights_input_hidden = beta * self.v_weights_input_hidden + (1.0 - beta) * grad_w_in
        self.v_bias_hidden = beta * self.v_bias_hidden + (1.0 - beta) * grad_b_in

        self.weights_hidden_output += self.v_weights_hidden_output * learning_rate
        self.bias_output += self.v_bias_output * learning_rate
        self.weights_input_hidden += self.v_weights_input_hidden * learning_rate
        self.bias_hidden += self.v_bias_hidden * learning_rate

    def train(
        self, X, y, X_val, y_val, epochs, learning_rate,
        batch_size=128, patience=3, print_every=1,
    ):
        history = []
        best_val_loss = float("inf")
        best_epoch = 1
        best_weights = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.total_reward = 0.0
            self.total_penalty = 0.0

            order = np.random.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                self.feed_forward(X[idx])
                self.backward(X[idx], y[idx], learning_rate)

            # sample training rows for faster epoch-level reporting
            sample_size = min(2000, len(X))
            sample_idx = order[:sample_size]
            train_pred = self.feed_forward(X[sample_idx])
            train_correct, _ = self.reward_penalty(train_pred, y[sample_idx])
            train_active = (
                y[sample_idx]
                .reshape(-1, self.seq_len, self.vocab_size)
                .argmax(axis=2) != STOI[PAD]
            )
            train_acc = float(train_correct.sum() / max(train_active.sum(), 1))

            val_pred = self.feed_forward(X_val)
            val_pred_3d = val_pred.reshape(-1, self.seq_len, self.vocab_size)
            val_true_3d = y_val.reshape(-1, self.seq_len, self.vocab_size)
            val_true_idx = val_true_3d.argmax(axis=2)
            val_active = val_true_idx != STOI[PAD]
            # exclude padding positions from validation loss and accuracy
            token_losses = -np.sum(
                val_true_3d * np.log(val_pred_3d + 1e-9), axis=2
            )
            per_example_loss = np.sum(token_losses * val_active, axis=1) / np.maximum(
                val_active.sum(axis=1), 1
            )
            val_loss = float(np.mean(per_example_loss))
            val_correct, _ = self.reward_penalty(val_pred, y_val)
            val_acc = float(val_correct.sum() / max(val_active.sum(), 1))

            history.append({
                "epoch": epoch,
                "train_char_acc": train_acc,
                "val_loss": val_loss,
                "val_char_acc": val_acc,
                "reward": self.total_reward,
                "penalty": self.total_penalty,
            })

            if epoch % print_every == 0 or epoch == epochs:
                print(
                    f"epoch: {epoch:2d}/{epochs:2d} | "
                    f"val loss: {val_loss:.4f} | "
                    f"train acc: {train_acc:.3f} | "
                    f"val acc: {val_acc:.3f} | "
                    f"reward: {self.total_reward:6.0f} | "
                    f"penalty: {self.total_penalty:6.0f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_weights = (
                    self.weights_input_hidden.copy(),
                    self.bias_hidden.copy(),
                    self.weights_hidden_output.copy(),
                    self.bias_output.copy(),
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience is not None and patience_counter >= patience:
                    print(
                        f"early stopping at epoch {epoch} "
                        f"(restoring best weights from epoch {best_epoch})"
                    )
                    break

        if best_weights is not None:
            (
                self.weights_input_hidden,
                self.bias_hidden,
                self.weights_hidden_output,
                self.bias_output,
            ) = best_weights

        return history

    def predict_text(self, x):
        probs = self.feed_forward(x).reshape(self.seq_len, self.vocab_size).copy()
        probs[:, STOI[BOS]] = 0.0
        indices = probs.argmax(axis=1)
        chars = []
        for index in indices:
            if index in (STOI[PAD], STOI[EOS]):
                break
            chars.append(ITOS[int(index)])
        return "".join(chars).rstrip()

    def save(self, path):
        """save model weights and dimensions for later inference"""
        np.savez_compressed(
            path,
            weights_input_hidden=self.weights_input_hidden,
            bias_hidden=self.bias_hidden,
            weights_hidden_output=self.weights_hidden_output,
            bias_output=self.bias_output,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            reward_points=self.reward_points,
            penalty_points=self.penalty_points,
            momentum=self.momentum,
        )

    def decode_with_feedback(self, x, word_set, rounds=20):
        """improve text predictions using a word list"""
        probs = self.feed_forward(x).reshape(self.seq_len, self.vocab_size).copy()
        probs[:, STOI[BOS]] = 0.0
        ranking = np.argsort(-probs, axis=1)
        rank = np.zeros(self.seq_len, dtype=int)

        def current_text():
            return "".join(ITOS[ranking[i, rank[i]]] for i in range(self.seq_len))

        def split_words(text):
            cleaned = text.replace(PAD, " ").replace(EOS, " ")
            return cleaned.split(" ")

        def score_text(text):
            score = 0.0
            for w in split_words(text):
                w_strip = w.strip()
                if not w_strip:
                    continue
                score += self.reward_points if w_strip in word_set else -self.penalty_points
            return score

        best_text = current_text()
        best_score = score_text(best_text)

        for _ in range(rounds):
            text = current_text()
            bad_positions, pos = [], 0
            for w in text.replace(PAD, " ").replace(EOS, " ").split(" "):
                if w and w not in word_set:
                    bad_positions.extend(range(pos, pos + len(w)))
                pos += len(w) + 1

            if not bad_positions:
                break

            # find the least confident character in penalised words
            conf = [probs[i, ranking[i, rank[i]]] for i in bad_positions]
            worst = bad_positions[int(np.argmin(conf))]
            rank[worst] = min(rank[worst] + 1, self.vocab_size - 1)

            new_text = current_text()
            new_score = score_text(new_text)
            if new_score > best_score:
                best_text, best_score = new_text, new_score

        clean = best_text.split(PAD, 1)[0].split(EOS, 1)[0].rstrip()
        return clean, best_score


# TRAINING MODEL ON EACH CIPHER ->
def run_cipher(cipher, args, index, all_sentences):
    np.random.seed(args.seed)
    random.seed(args.seed)

    result_dir = RESULTS_DIR / cipher
    result_dir.mkdir(parents=True, exist_ok=True)

    # data loading + this cipher's own preprocessing
    df = PREPROCESSORS[cipher](load_cipher_file(cipher), index)
    dropped = df.attrs.get("dropped", 0)
    test_set, valid_set = sentence_splits(all_sentences, args.seed)
    df = add_split(df, test_set, valid_set, set(all_sentences))

    if args.limit:
        df = df.sample(frac=1, random_state=args.seed).groupby("split").head(args.limit)

    train = df[df.split == "train"]
    valid = df[df.split == "valid"]
    test = df[df.split == "test"]
    if train.empty or valid.empty or test.empty:
        raise ValueError(
            f"{cipher} needs non-empty train, validation, and test splits; "
            "check the data or increase --limit"
        )

    max_input_len = int(df.ciphertext.str.len().max())
    max_target_len = int(df.plaintext.str.len().max()) + 1
    seq_len = args.seq_len or max(max_input_len, max_target_len)
    if seq_len < max(max_input_len, max_target_len):
        raise ValueError(
            f"--seq-len {seq_len} is too short for {cipher}: "
            f"need at least {max(max_input_len, max_target_len)}"
        )
    input_size = seq_len * V
    output_size = seq_len * V

    X_train = one_hot_encode(list(train.ciphertext), seq_len)
    y_train = encode_targets(list(train.plaintext), seq_len)

    X_val = one_hot_encode(list(valid.ciphertext), seq_len)
    y_val = encode_targets(list(valid.plaintext), seq_len)

    X_test = one_hot_encode(list(test.ciphertext), seq_len)

    model = ClassicalMLP(
        input_size=input_size,
        hidden_size=args.hid,
        output_size=output_size,
        seq_len=seq_len,
        vocab_size=V,
        reward_points=args.reward,
        penalty_points=args.penalty,
        momentum=args.momentum,
    )

    start_time = time.time()
    history = model.train(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    minutes = (time.time() - start_time) / 60.0

    model.save(result_dir / f"{cipher}_classicalMLP.npz")
    pd.DataFrame(history).to_csv(result_dir / "history.csv", index=False)

    # optional word-list feedback uses only words from the training split
    word_set = set(w for p in train.plaintext for w in p.split(" ") if w)

    truths = list(test.plaintext)
    test_ciphers = list(test.ciphertext)
    preds = []
    feedback_preds = []

    for i in range(len(test)):
        xi = X_test[i:i + 1]
        p_text = model.predict_text(xi)
        preds.append(p_text)
        if args.eval_feedback:
            fb_text, _ = model.decode_with_feedback(xi, word_set)
            feedback_preds.append(fb_text)

    chosen_preds = feedback_preds if args.eval_feedback else preds
    exact = float(np.mean([p == t for p, t in zip(chosen_preds, truths)]))
    cer = float(np.mean([
        Levenshtein.distance(p, t) / max(len(t), 1)
        for p, t in zip(chosen_preds, truths)
    ]))

    pred_df = pd.DataFrame({
        "ciphertext": test_ciphers,
        "true": truths,
        "pred": chosen_preds,
        "exact": [p == t for p, t in zip(chosen_preds, truths)],
    })
    if args.eval_feedback:
        pred_df["pred_base"] = preds
    pred_df.to_csv(result_dir / "predictions.csv", index=False)

    best_epoch_idx = int(np.argmin([h["val_loss"] for h in history])) if history else 0
    best_val_acc = round(history[best_epoch_idx]["val_char_acc"], 4) if history else 0.0

    return {
        "cipher": cipher,
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "dropped": dropped,
        "epochs_run": len(history),
        "best_val_acc": best_val_acc,
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
        raw = input("enter number(s), e.g. 1 or 2,4: ").replace(",", " ").split()
        try:
            picks = [int(x) for x in raw]
        except ValueError:
            picks = []
        if picks and all(1 <= p <= 6 for p in picks):
            return names if 6 in picks else [names[p - 1] for p in dict.fromkeys(picks)]
        print("invalid choice... try again.")


# main ->
def main():
    parser = argparse.ArgumentParser(
        description="train a classical MLP for cipher decryption."
    )
    parser.add_argument("--ciphers", nargs="+", choices=list(CIPHER_FILES),
                        help="ciphers to train; if omitted you will be asked")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hid", type=int, default=512, help="hidden layer size")
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum coefficient")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="sequence length; inferred from the data if omitted")
    parser.add_argument("--reward", type=float, default=1.0,
                        help="reward points for correct char")
    parser.add_argument("--penalty", type=float, default=1.0,
                        help="penalty points for incorrect char")
    parser.add_argument(
        "--eval-feedback",
        action="store_true",
        help="run dictionary-based feedback decoding on test",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="rows per split for quick smoke tests")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hid < 1:
        parser.error("--epochs, --batch-size, and --hid must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if not 0 <= args.momentum < 1:
        parser.error("--momentum must be in [0, 1)")
    if args.patience is not None and args.patience < 1:
        parser.error("--patience must be positive")
    if args.seq_len is not None and args.seq_len < 1:
        parser.error("--seq-len must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.reward < 0 or args.penalty < 0:
        parser.error("--reward and --penalty must be non-negative")

    ciphers = args.ciphers or ask_user_for_ciphers()

    missing = [
        path.name
        for path in [SOURCE_FILE] + [DATA_DIR / CIPHER_FILES[c] for c in ciphers]
        if not path.exists()
    ]
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
    table = tabulate([[r[h] for h in headers] for r in rows], headers=headers, tablefmt="github")
    print("\n" + table)

    summary_path = RESULTS_DIR / "combined_summary.md"
    summary_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nsaved summary to {summary_path}")


if __name__ == "__main__":
    main()
