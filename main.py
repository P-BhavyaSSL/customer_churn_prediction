import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
df = pd.read_csv('Customer-Churn-Records.csv')
# print(df.head())
# print(df.info())
# print(df.shape)
# print(df.isnull().sum())
# print(df.duplicated().sum())
# df = df.drop_duplicates()
# print(df.shape)
df = df.drop(
    columns=["RowNumber", "CustomerId", "Surname"]
)
# print(df.shape)
# print(df["Exited"].value_counts())

# print("\nTarget percentage:")

# print(
#     df["Exited"]
#     .value_counts(normalize=True)
#     .mul(100)
# )

# categorical_columns_original = [
#     "Geography",
#     "Gender",
#     "Card Type"
# ]

# print("\n================ CATEGORICAL VALUES ================\n")

# for column in categorical_columns_original:
#     if column in df.columns:
#         print(f"\n{column}:")
#         print(df[column].unique())

X = df.drop("Exited", axis=1)
print(X.head())
y = df["Exited"]
print(y.head())
numerical_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Complain",
    "Satisfaction Score",
    "Point Earned"
]

categorical_features = [
    "Geography",
    "Gender",
    "Card Type"
]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

#numerical processing and categorical processing pipelines

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

#model creation

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

#training the model

model.fit(X_train, y_train)

#prediction

y_pred = model.predict(X_test)

# Probability of class 1 = probability of churn
y_probability = model.predict_proba(X_test)[:, 1]

#model scores

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_probability)


print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC-AUC  :", roc_auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#churn risk logic

def churn_risk(probability):

    if probability < 0.30:
        return "Low"

    elif probability < 0.70:
        return "Medium"

    else:
        return "High"


risk = [churn_risk(p) for p in y_probability]

#prediction result

results = X_test.copy()

results["Actual"] = y_test.values
results["Predicted"] = y_pred
results["Churn Probability"] = y_probability
results["Risk"] = risk

print("\nPredictions:")
print(results.head(10))
print(results.shape)