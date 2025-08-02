import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Fetch Adult Income dataset (Census data with gender, race, salary)
adult = fetch_openml('adult', version=2, as_frame=True)
df = adult.frame.dropna()

# 2. Prepare data: Predict 'Salary' using a simple decision tree
X = pd.get_dummies(df.drop('class', axis=1), drop_first=True)
y = (df['class'] == '>50K').astype(int) # Binary: High income?

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# 3. Audit: Check error rate by gender group
test_idx = X_test.index
test_gender =df.loc[test_idx, "sex"]
results = pd.DataFrame({'actual': y_test, 'predicted': y_pred, 'gender': test_gender.values})

male_errors = (results['gender'] == 'Male') & (results['actual'] != results['predicted'])
female_errors = (results['gender'] == 'Female') & (results['actual'] != results['predicted'])

male_error_rate = male_errors.sum() / (results['gender'] == 'Male').sum()
female_error_rate = female_errors.sum() / (results['gender'] == 'Female').sum()

print(f"Male Error Rate: {male_error_rate:.3f}")
print(f"Female Error Rate: {female_error_rate:.3f}")

# Ethical note: Large gaps highlight bias-repeat for other groups (race, age) for a full audit.
