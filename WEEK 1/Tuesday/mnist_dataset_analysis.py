import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# 1. Download MNIST dataset (auto-cached by sciki-learn)
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)

# 2. Quick data summary
print("MNIST Dataset Size:", X.shape)
print("Labels (Digits 0-9):", sorted(set(y)))

# 3. Distribution: How many images for each digit?
digit_counts = pd.Series(y).value_counts().sort_index()
print("\nLabel Distribution:\n", digit_counts)

# 4. Visualize: Example image and histogram of digit counts
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(X.iloc[0].values.reshape(28, 28), cmap="gray")
plt.title(f"Sample Digit (Label: {y[0]})")
plt.axis("off")

plt.subplot(1, 2, 2)
digit_counts.plot(kind="bar")
plt.title("Digit Distribution")
plt.xlabel("Digit")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("mnist_insights.png")
plt.show()

# 5. Ethical note: Remember, skewed data (here, balanced!) can bias models in real-life AI
