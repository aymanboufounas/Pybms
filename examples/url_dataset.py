"""Use the public example CSV directly after installing from GitHub."""

import pybms as pb

url = "https://raw.githubusercontent.com/aymanboufounas/Pybms/main/examples/data/demo.csv"
df = pb.from_url(url)
result = pb.auto_train(df, target="passed", trials=2, cv=2)
print(result.metrics)
