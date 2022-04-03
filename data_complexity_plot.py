import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("dcomplexity.log")
data.columns = [a.strip() for a in data.columns]

names = ["KNN", "LR", "LDA", "SVM", "DT", "GP", "DNN"]
size = data["size"]

r = data.iloc[:, :7]

fig, ax = plt.subplots(1, 1, figsize = (8, 4))

cls = ["k", "b", "r", "g", "grey", "purple", "c"]
for i in range(7):
    ax.loglog(size, 1 - r.iloc[:, i], ".", label = names[i], color = cls[i], markersize = 4)

ax.set_xlabel("Dataset size")
ax.set_ylabel("Testing Set 1 - BAS")
ax.set_xlim([800, 28000])
plt.legend(loc="center left", bbox_to_anchor=(1,0.5))
plt.tight_layout()
plt.show()
