"""Run: python examples/tabular_quickstart.py"""

import pybms as pb

df = pb.sample_data("iris")
result = pb.auto_train(df, target="target", trials=2, cv=3)
print("Test metrics:", result.metrics)
print(result.leaderboard[["estimator", "cv_score", "cv_std"]])
print("Predictions:", result.predict(df.head()))
print("Setup:", pb.explain_model(result))
