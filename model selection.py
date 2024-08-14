import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import joblib

file_path = "C:/Users/Vaidehi Suryawanshi/Downloads/churn/archive/Churn_Modelling.csv"
data = pd.read_csv(file_path)

# Drop unnecessary columns
data = data.drop(['CustomerId', 'Surname', 'RowNumber'], axis=1)

# Convert categorical variables to numerical using Label Encoding
label_encoder = LabelEncoder()
data['Geography'] = label_encoder.fit_transform(data['Geography'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Separate features and target variable
X = data.drop('Exited', axis=1)
y = data['Exited']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
log_reg = LogisticRegression(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)

# Train and evaluate Logistic Regression
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log)
log_reg_roc_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
log_reg_report = classification_report(y_test, y_pred_log)
log_reg_conf_matrix = confusion_matrix(y_test, y_pred_log)

# Train and evaluate Random Forest
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_roc_auc = roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1])
rf_report = classification_report(y_test, y_pred_rf)
rf_conf_matrix = confusion_matrix(y_test, y_pred_rf)

# Train and evaluate Gradient Boosting
gb_clf.fit(X_train, y_train)
y_pred_gb = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, y_pred_gb)
gb_roc_auc = roc_auc_score(y_test, gb_clf.predict_proba(X_test)[:, 1])
gb_report = classification_report(y_test, y_pred_gb)
gb_conf_matrix = confusion_matrix(y_test, y_pred_gb)

# Print results for all models
print("\nLogistic Regression Accuracy:", log_reg_accuracy)
print("ROC AUC Score:", log_reg_roc_auc)
print("Classification Report:\n", log_reg_report)
print("Confusion Matrix:\n", log_reg_conf_matrix)

print("\nRandom Forest Accuracy:", rf_accuracy)
print("ROC AUC Score:", rf_roc_auc)
print("Classification Report:\n", rf_report)
print("Confusion Matrix:\n", rf_conf_matrix)

print("\nGradient Boosting Accuracy:", gb_accuracy)
print("ROC AUC Score:", gb_roc_auc)
print("Classification Report:\n", gb_report)
print("Confusion Matrix:\n", gb_conf_matrix)

# Choose the best model based on ROC AUC Score
best_model = None
best_score = -1
best_model_name = ""

if gb_roc_auc > best_score:
    best_model = gb_clf
    best_score = gb_roc_auc
    best_model_name = "Gradient Boosting"
elif rf_roc_auc > best_score:
    best_model = rf_clf
    best_score = rf_roc_auc
    best_model_name = "Random Forest"
elif log_reg_roc_auc > best_score:
    best_model = log_reg
    best_score = log_reg_roc_auc
    best_model_name = "Logistic Regression"

print(f"\nBest Model: {best_model_name}")
print(f"Best Model ROC AUC Score: {best_score}")

# Save the best model and scaler
joblib.dump(best_model, 'best_churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"\nThe best model ({best_model_name}) has been saved as 'best_churn_model.pkl'.")
print("The preprocessor has been saved as 'scaler.pkl'.")
