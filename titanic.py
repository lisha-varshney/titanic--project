import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
# Load Titanic dataset
df = pd.read_csv("titanic-dataset.csv")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
# Select features and target
features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]
X = df[features]
y = df["survived"]
# Numerical and categorical columns
numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]
categorical_features = ["sex","embarked"]
# Handle missing numerical values
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median"))])
# Handle missing categorical values
categorical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)])
# Create model
model = Pipeline(
    steps=[("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42))])
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,
    random_state=42,stratify=y)
# Train model
model.fit(X_train, y_train)
# Prediction
predictions = model.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:")
print(accuracy)
# Classification Report
print("\nClassification Report:")
print(classification_report(y_test,predictions))
# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test,predictions)
plt.title("Titanic Survival Prediction")
plt.show()