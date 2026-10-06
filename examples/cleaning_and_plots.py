"""Run with MPLBACKEND=Agg for noninteractive plotting."""

import matplotlib.pyplot as plt
import pybms as pb

df = pb.random_data(rows=100, columns=3, missing=0.1)
cleaned, report = pb.clean_data(df, report=True)
print(report)
print(pb.missing_report(df))
ax = pb.hist_plot(cleaned)
ax.figure.savefig("histogram.png", dpi=150)
plt.close(ax.figure)
print("Saved histogram.png")
