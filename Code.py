import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# Read data from CSV
df = pd.read_csv("data.csv")
X = df[["X1", "X2"]].values
y = df["y"].values
# Perform Logistic Regression
logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(X)
# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", alpha=0.8)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Logistic Regression")
plt.show()
