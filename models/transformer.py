from pathlib import Path

# PATH SETUP
BASE_DIR = Path(__file__).resolve().parent

DATA_FILES = {
    "caesar_path": BASE_DIR / "data" / "caesar_gutenberg_50k.csv",
    "columnar_path": BASE_DIR / "data" / "columnar_transposition_gutenberg_50k.csv",
    "hill_path": BASE_DIR / "data" / "hill_gutenberg_50k.csv",
    "monoalphabetic_path": BASE_DIR / "data" / "monoalphabetic_gutenberg_50k.csv",
    "vigenere_path": BASE_DIR / "data" / "vigenere_gutenberg_50k.csv",
}
missing = [name for name, path in DATA_FILES.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"data CSV not found for: {', '.join(missing)}")