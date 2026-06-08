"""
Baseline solution: TF-IDF + Logistic Regression for Jigsaw Unintended Bias.

Produces submission.csv with columns: id, prediction.
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


data_dir = Path("data")

train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")

text_col = "comment_text"
target_col = "target"

X_train = train[text_col].fillna("")
y_train = (train[target_col] >= 0.5).astype(int)
X_test = test[text_col].fillna("")

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=200000,
        sublinear_tf=True,
        strip_accents="unicode",
    )),
    ("clf", LogisticRegression(
        C=4.0,
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
    )),
])

pipe.fit(X_train, y_train)
pred = pipe.predict_proba(X_test)[:, 1]

sub = pd.DataFrame({"id": test["id"], "prediction": pred.clip(0.0, 1.0)})
sub.to_csv("submission.csv", index=False)
print(f"submission.csv written ({len(sub)} rows)")
