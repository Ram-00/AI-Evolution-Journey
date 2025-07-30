from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load digits dataset (0-9 grayscale images)
digits = load_digits()
X, y = digits.data, digits.target

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a simple Logistic regression classifier
model = LogisticRegression(max_iter=200, solver='liblinear')
model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.3f}")

# 5. Ethical reflection
print("Note: Even high accuracy models can inherit bias from training data-always check who benefits and who might be left out.")
