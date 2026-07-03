from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


def main() -> None:
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_scaled, y_train)
    print(classification_report(y_test, model.predict(X_test_scaled), target_names=data.target_names))

    output = Path("outputs")
    output.mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, output / "gaussian_nb_pipeline.joblib")


if __name__ == "__main__":
    main()
