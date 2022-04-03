import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

data = pd.read_csv(Path("accident_classification/operation_accidents.csv"))
data["accident"] = [a.title() for a in data["accident"]]

ax = data["accident"].value_counts().plot(kind = "barh",
        figsize = (8, 3), color= "k")
ax.set_xlabel("Freq.")
plt.tight_layout()
plt.show()
