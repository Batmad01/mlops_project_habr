import os
import sys

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "train"]
    os.system(f"uv run python -m habr_articles_classifier.train {' '.join(args)}")
