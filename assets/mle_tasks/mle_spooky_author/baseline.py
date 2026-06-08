"""
Baseline solution: TF-IDF + Logistic Regression for spooky-author-identification.

Produces submission.csv with columns: id, EAP, HPL, MWS (probability per author).
Run evaluation: bash run.sh result.json
"""
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

data_dir = Path("data")

train = pd.read_csv(data_dir / "train.csv")
test  = pd.read_csv(data_dir / "test.csv")

X_train = train["text"]
y_train = train["author"]
X_test  = test["text"]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
    )),
])

pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)
classes = pipe.classes_

sub = pd.DataFrame(proba, columns=classes)
sub.insert(0, "id", test["id"])
sub.to_csv("submission.csv", index=False)
print(f"submission.csv written ({len(sub)} rows, columns: {list(sub.columns)})")
