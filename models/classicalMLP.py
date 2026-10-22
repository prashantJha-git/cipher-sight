import numpy as np
import pandas as pd
import time
import random
import argparse
from pathlib import Path
from rapidfuzz.distance import Levenshtein
from tabulate import tabulate

# PATH SETUP (robust to running from repo root or models directory)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" if (BASE_DIR / "data").exists() else BASE_DIR.parent / "data"
RESULTS_DIR = BASE_DIR / "model_results" / "classicalMLP" if (BASE_DIR / "model_results").parent.exists() else BASE_DIR.parent / "model_results" / "classicalMLP"

CIPHER_FILES = {
    "caesar": "caesar_gutenberg_50k.csv",
    "columnar": "columnar_transposition_gutenberg_50k.csv",
    "hill": "hill_gutenberg_50k.csv",
    "monoalphabetic": "monoalphabetic_gutenberg_50k.csv",
    "vigenere": "vigenere_gutenberg_50k.csv",
}
# Reference sentences needed to recover spaces and strip padding for columnar/hill
SOURCE_FILE = DATA_DIR / "random_sentences_gutenberg_50k.txt"

# Unified vocabulary aligned with LSTM.py
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
VOCAB = [PAD, BOS, EOS, " "] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for c, i in STOI.items()}
V = len(VOCAB)


# DATA LOADING & CIPHER PREPROCESSING (shared standard with LSTM.py)
def load_source_sentences():
    lines = [l.strip() for l in SOURCE_FILE.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l]


def build_nospace_lookup(sentences):
    """Building stripped sentence lookup to restore spaces"""
    index = {}
    for s in sentences:
        index.setdefault(s.replace(" ", ""), set()).add(s)
    return index


def unambiguous_sentences(sentences):
    """Keep sentences whose no-space form is unique"""
    index = build_nospace_lookup(sentences)
    return sorted({s for hits in index.values() if len(hits) == 1 for s in hits})


def load_cipher_file(cipher):
    return pd.read_csv(DATA_DIR / CIPHER_FILES[cipher], dtype=str, keep_default_na=False)


def sentence_splits(sentences, seed):
    """Consistent sentence split across all models (80% train, 10% valid, 10% test)"""
    uniq = sorted(set(sentences))
    random.Random(seed).shuffle(uniq)
    n = len(uniq)
    n_test = n_valid = int(0.1 * n)
    test = set(uniq[:n_test])
    valid = set(uniq[n_test:n_test + n_valid])
    return test, valid


def recover_target(pt, index):
    """Restore unique spaced sentence, ignoring trailing X padding"""
    for k in range(0, 9):
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
    """Ciphers with already spaced plaintext: strip ciphertext spaces"""
    rows = [(ct.replace(" ", ""), pt) for ct, pt in zip(df.ciphertext, df.plaintext)]
    return _finish(rows, 0)


def _unspaced_plaintext_preprocessing(df, index):
    """Ciphers where plaintext lost spaces / got X padding: restore true sentence"""
    rows, dropped = [], 0
    for ct, pt in zip(df.ciphertext, df.plaintext):
        target = recover_target(pt, index)
        if target is None:
            dropped += 1
            continue
        rows.append((ct.replace(" ", ""), target))
    return _finish(rows, dropped)


PREPROCESSORS = {
    "caesar": lambda df, idx: _spaced_plaintext_preprocessing(df),
    "columnar": lambda df, idx: _unspaced_plaintext_preprocessing(df, idx),
    "hill": lambda df, idx: _unspaced_plaintext_preprocessing(df, idx),
    "monoalphabetic": lambda df, idx: _spaced_plaintext_preprocessing(df),
    "vigenere": lambda df, idx: _spaced_plaintext_preprocessing(df),
}


def add_split(df, test, valid, allowed):
    df = df[df.plaintext.isin(allowed)]
    split = np.where(df.plaintext.isin(test), "test", np.where(df.plaintext.isin(valid), "valid", "train"))
    return df.assign(split=split)


# ONE-HOT ENCODING FOR CLASSICAL MLP
def one_hot_encode(texts, seq_len):
    """Encodes strings into 2D one-hot vectors (N, seq_len * V)"""
    arr = np.zeros((len(texts), seq_len, V), dtype=np.float32)
    for i, t in enumerate(texts):
        chars = list(t[:seq_len])
        for j, c in enumerate(chars):
            idx = STOI.get(c, STOI[PAD])
            arr[i, j, idx] = 1.0
        # pad remaining length
        for j in range(len(chars), seq_len):
            arr[i, j, STOI[PAD]] = 1.0
    return arr.reshape(len(texts), -1)


# CLASSICAL NEURAL NETWORK (ENHANCED MLP WITH RELU, MOMENTUM & EARLY STOPPING)
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, seq_len, vocab_size,
                 reward_points=1.0, penalty_points=1.0, momentum=0.9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # REWARD / PENALTY SETTINGS
        self.reward_points = reward_points
        self.penalty_points = penalty_points
        self.total_reward = 0.0
        self.total_penalty = 0.0

        # He initialization optimal for ReLU / LeakyReLU
        self.weights_input_hidden = np.random.randn(
            self.input_size, self.hidden_size
        ).astype(np.float32) * np.sqrt(2.0 / self.input_size)
        self.bias_hidden = np.zeros((1, self.hidden_size), dtype=np.float32)

        self.weights_hidden_output = np.random.randn(
            self.hidden_size, self.output_size
        ).astype(np.float32) * np.sqrt(2.0 / self.hidden_size)
        self.bias_output = np.zeros((1, self.output_size), dtype=np.float32)

        # MOMENTUM OPTIMIZER BUFFERS (Feature 2)
        self.momentum = momentum
        self.v_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.v_bias_hidden = np.zeros_like(self.bias_hidden)
        self.v_weights_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.v_bias_output = np.zeros_like(self.bias_output)

    # LEAKY RELU ACTIVATION & DERIVATIVE (Feature 1: Eliminates vanishing gradients)
    def relu(self, x):
        return np.maximum(0.01 * x, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1.0, 0.01)

    def softmax_per_char(self, x):
        """Softmax over vocabulary dimension for each character position"""
        n = x.shape[0]
        x3d = x.reshape(n, self.seq_len, self.vocab_size)
        x_max = np.max(x3d, axis=2, keepdims=True)
        exp_x = np.exp(x3d - x_max)
        probs = exp_x / (np.sum(exp_x, axis=2, keepdims=True) + 1e-9)
        return probs.reshape(n, -1)

    # FEED FORWARD
    def feedForward(self, x):
        self.hidden_activation = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_activation)

        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.softmax_per_char(self.output_activation)
        return self.predicted_output

    # REWARD AND PENALTY SYSTEM
    def rewardPenalty(self, predicted_output, y):
        """
        Compare argmax character against true character:
          correct char -> +reward_points
          wrong char   -> -penalty_points
        """
        n = predicted_output.shape[0]
        pred_idx = predicted_output.reshape(n, self.seq_len, self.vocab_size).argmax(axis=2)
        true_idx = y.reshape(n, self.seq_len, self.vocab_size).argmax(axis=2)

        correct = (pred_idx == true_idx)
        points = np.where(correct, self.reward_points, -self.penalty_points)
        return correct, points

    # BACKWARD PASS WITH REWARD/PENALTY & MOMENTUM (Feature 2)
    def backward(self, X, y, learning_rate):
        n = X.shape[0]

        # 1) Score predictions for reward & penalty
        correct, points = self.rewardPenalty(self.predicted_output, y)
        self.total_reward += float(points[points > 0].sum())
        self.total_penalty += float(-points[points < 0].sum())

        # 2) Error multiplier per character position:
        #    Reward  -> smaller gradient adjustment
        #    Penalty -> stronger gradient push
        multiplier = np.where(
            correct,
            1.0 / (1.0 + self.reward_points),
            1.0 + self.penalty_points,
        )  # (n, seq_len)
        multiplier = np.repeat(multiplier, self.vocab_size, axis=1)  # (n, seq_len * V)

        # Cross-entropy with Softmax gradient: (y - p) * multiplier
        output_error = (y - self.predicted_output) * multiplier

        # Backpropagation through hidden layer using ReLU derivative
        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.relu_derivative(self.hidden_activation)

        # Gradient computation
        scale = 1.0 / n
        grad_w_out = np.dot(self.hidden_output.T, output_error) * scale
        grad_b_out = np.sum(output_error, axis=0, keepdims=True) * scale
        grad_w_in = np.dot(X.T, hidden_delta) * scale
        grad_b_in = np.sum(hidden_delta, axis=0, keepdims=True) * scale

        # Momentum velocity updates: v = beta * v + (1 - beta) * grad
        beta = self.momentum
        self.v_weights_hidden_output = beta * self.v_weights_hidden_output + (1.0 - beta) * grad_w_out
        self.v_bias_output = beta * self.v_bias_output + (1.0 - beta) * grad_b_out
        self.v_weights_input_hidden = beta * self.v_weights_input_hidden + (1.0 - beta) * grad_w_in
        self.v_bias_hidden = beta * self.v_bias_hidden + (1.0 - beta) * grad_b_in

        # Weight updates with momentum velocity
        self.weights_hidden_output += self.v_weights_hidden_output * learning_rate
        self.bias_output += self.v_bias_output * learning_rate
        self.weights_input_hidden += self.v_weights_input_hidden * learning_rate
        self.bias_hidden += self.v_bias_hidden * learning_rate

    # TRAINING LOOP WITH EARLY STOPPING & BEST WEIGHT RESTORATION (Feature 3)
    def train(self, X, y, X_val, y_val, epochs, learning_rate, batch_size=128, patience=3, print_every=1):
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
                self.feedForward(X[idx])
                self.backward(X[idx], y[idx], learning_rate)

            # Fast evaluation: sample subset of training set for logging
            sample_size = min(2000, len(X))
            sample_idx = order[:sample_size]
            train_pred = self.feedForward(X[sample_idx])
            train_correct, _ = self.rewardPenalty(train_pred, y[sample_idx])
            train_acc = float(train_correct.mean())

            # Validation evaluation
            val_pred = self.feedForward(X_val)
            val_loss = float(-np.mean(np.sum(y_val * np.log(val_pred + 1e-9), axis=1)))
            val_correct, _ = self.rewardPenalty(val_pred, y_val)
            val_acc = float(val_correct.mean())

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
                    f"Epoch {epoch:2d}/{epochs:2d} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc:.3f} | "
                    f"Val Acc: {val_acc:.3f} | "
                    f"Reward: {self.total_reward:6.0f} | "
                    f"Penalty: {self.total_penalty:6.0f}"
                )

            # Early stopping check (matching LSTM.py EarlyStopping callback)
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
                    print(f"Early stopping at epoch {epoch} (restoring best weights from epoch {best_epoch})")
                    break

        # Restore best weights
        if best_weights is not None:
            (
                self.weights_input_hidden,
                self.bias_hidden,
                self.weights_hidden_output,
                self.bias_output,
            ) = best_weights

        return history

    # ARGMAX PREDICTION PER CHARACTER
    def predictText(self, x):
        probs = self.feedForward(x).reshape(self.seq_len, self.vocab_size)
        indices = probs.argmax(axis=1)
        chars = [ITOS[i] for i in indices]
        # Drop PAD tokens for final sentence output
        text = "".join(chars)
        return text.replace(PAD, "").replace(EOS, "").rstrip()

    # TEST-TIME DECODING WITH WORD-LIST FEEDBACK
    def decodeWithFeedback(self, x, word_set, rounds=20):
        """
        Test-time search: uses vocabulary word list as an English proxy.
        Characters in words not found in word_set are flipped to their next best candidate.
        """
        probs = self.feedForward(x).reshape(self.seq_len, self.vocab_size)
        ranking = np.argsort(-probs, axis=1)         # highest probability candidate first
        rank = np.zeros(self.seq_len, dtype=int)     # index of candidate currently used

        def currentText():
            return "".join(ITOS[ranking[i, rank[i]]] for i in range(self.seq_len))

        def splitWords(text):
            cleaned = text.replace(PAD, " ").replace(EOS, " ")
            return cleaned.split(" ")

        def scoreText(text):
            score = 0.0
            for w in splitWords(text):
                w_strip = w.strip()
                if not w_strip:
                    continue
                score += self.reward_points if w_strip in word_set else -self.penalty_points
            return score

        best_text = currentText()
        best_score = scoreText(best_text)

        for _ in range(rounds):
            text = currentText()
            bad_positions, pos = [], 0
            for w in text.replace(PAD, " ").replace(EOS, " ").split(" "):
                if w and w not in word_set:
                    bad_positions.extend(range(pos, pos + len(w)))
                pos += len(w) + 1

            if not bad_positions:
                break

            # Find least confident character in penalised words
            conf = [probs[i, ranking[i, rank[i]]] for i in bad_positions]
            worst = bad_positions[int(np.argmin(conf))]
            rank[worst] = min(rank[worst] + 1, self.vocab_size - 1)

            new_text = currentText()
            new_score = scoreText(new_text)
            if new_score > best_score:
                best_text, best_score = new_text, new_score

        clean = best_text.replace(PAD, "").replace(EOS, "").rstrip()
        return clean, best_score


# TRAIN & EVALUATE ON A SINGLE CIPHER
def run_cipher(cipher, args, index, all_sentences):
    np.random.seed(args.seed)
    random.seed(args.seed)

    result_dir = RESULTS_DIR / cipher
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data loading and cipher preprocessing
    df = PREPROCESSORS[cipher](load_cipher_file(cipher), index)
    dropped = df.attrs.get("dropped", 0)
    test_set, valid_set = sentence_splits(all_sentences, args.seed)
    df = add_split(df, test_set, valid_set, set(all_sentences))

    if args.limit:
        df = df.sample(frac=1, random_state=args.seed).groupby("split").head(args.limit)

    train = df[df.split == "train"].copy()
    valid = df[df.split == "valid"].copy()
    test = df[df.split == "test"].copy()

    seq_len = args.seq_len or min(64, int(max(df.ciphertext.str.len().max(), df.plaintext.str.len().max())))
    input_size = seq_len * V
    output_size = seq_len * V

    # 2) One-hot encoding
    X_train = one_hot_encode(list(train.ciphertext), seq_len)
    y_train = one_hot_encode(list(train.plaintext), seq_len)

    X_val = one_hot_encode(list(valid.ciphertext), seq_len)
    y_val = one_hot_encode(list(valid.plaintext), seq_len)

    X_test = one_hot_encode(list(test.ciphertext), seq_len)

    # 3) Build model
    model = NeuralNetwork(
        input_size=input_size,
        hidden_size=args.hid,
        output_size=output_size,
        seq_len=seq_len,
        vocab_size=V,
        reward_points=args.reward,
        penalty_points=args.penalty,
        momentum=args.momentum,
    )

    # 4) Training with Early Stopping
    start_time = time.time()
    history = model.train(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    minutes = (time.time() - start_time) / 60.0

    # Save training history
    pd.DataFrame(history).to_csv(result_dir / "history.csv", index=False)

    # 5) Testing: Plain argmax vs Feedback decoding
    word_set = set(w for p in train.plaintext for w in p.split(" ") if w)

    truths = list(test.plaintext)
    test_ciphers = list(test.ciphertext)
    preds = []
    feedback_preds = []

    for i in range(len(test)):
        xi = X_test[i:i + 1]
        p_text = model.predictText(xi)
        preds.append(p_text)
        if args.eval_feedback:
            fb_text, _ = model.decodeWithFeedback(xi, word_set)
            feedback_preds.append(fb_text)

    chosen_preds = feedback_preds if args.eval_feedback else preds
    exact = float(np.mean([p == t for p, t in zip(chosen_preds, truths)]))
    cer = float(np.mean([Levenshtein.distance(p, t) / max(len(t), 1) for p, t in zip(chosen_preds, truths)]))

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
    """Interactive menu matching LSTM.py"""
    names = list(CIPHER_FILES)
    print("Which cipher do you want to train on?")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    print("  6. all (one after another, each with its own fresh model)")
    while True:
        raw = input("Enter number(s), e.g. 1 or 2,4: ").replace(",", " ").split()
        try:
            picks = [int(x) for x in raw]
        except ValueError:
            picks = []
        if picks and all(1 <= p <= 6 for p in picks):
            return names if 6 in picks else [names[p - 1] for p in dict.fromkeys(picks)]
        print("Invalid choice, try again.")


def main():
    parser = argparse.ArgumentParser(description="Classical MLP with Reward/Penalty Cryptanalysis")
    parser.add_argument("--ciphers", nargs="+", choices=list(CIPHER_FILES),
                        help="Ciphers to train; if omitted you will be asked")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hid", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum coefficient (default: 0.9)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (default: 3, matching LSTM)")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length cut/pad")
    parser.add_argument("--reward", type=float, default=1.0, help="Reward points for correct char")
    parser.add_argument("--penalty", type=float, default=1.0, help="Penalty points for incorrect char")
    parser.add_argument("--eval-feedback", action="store_true", help="Run dictionary-based feedback decoding on test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Rows per split for quick smoke tests")
    args = parser.parse_args()

    ciphers = args.ciphers or ask_user_for_ciphers()

    missing = [p.name for p in [SOURCE_FILE] + [DATA_DIR / CIPHER_FILES[c] for c in ciphers] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Data file(s) not found in data/: {', '.join(missing)}")

    all_sentences = unambiguous_sentences(load_source_sentences())
    index = build_nospace_lookup(all_sentences)

    rows = []
    for cipher in ciphers:
        print(f"\n===== Classical MLP: {cipher} =====")
        rows.append(run_cipher(cipher, args, index, all_sentences))

    headers = ["cipher", "train", "valid", "test", "dropped", "epochs_run",
               "best_val_acc", "test_exact", "test_cer", "minutes"]
    table = tabulate([[r[h] for h in headers] for r in rows], headers=headers, tablefmt="github")
    print("\n" + table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "combined_summary.md"
    summary_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
