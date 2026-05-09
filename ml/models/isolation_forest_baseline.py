from __future__ import annotations

from sklearn.ensemble import IsolationForest


def build_isolation_forest(random_state: int) -> IsolationForest:
    return IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )

