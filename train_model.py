# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# 📁 Ensure output folder exists
os.makedirs("trained_data", exist_ok=True)

# 📥 Load student grades dataset
df = pd.read_csv('csv/Students_Grading_Dataset.csv')

# ✅ Define pass/fail based on grade
# A, B, C → pass | D, E, F → fail
passing_grades = ['A', 'B', 'C']
df['label'] = df['Grade'].apply(lambda x: 'pass' if x in passing_grades else 'fail')

# 🔤 Encode target labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])  # pass=1, fail=0

# 🎯 Define input features
features = ['Total_Score', 'Midterm_Score', 'Final_Score', 'Projects_Score', 'Attendance (%)']
X = df[features]
y = df['label_encoded']

# 🧪 Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train logistic regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# ✅ Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy on test set: {accuracy:.2%}")

# 💾 Save model and label encoder
joblib.dump(model, "trained_data/model_cls.pkl")
joblib.dump(label_encoder, "trained_data/label_encoder.pkl")
