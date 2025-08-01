import pandas as pd
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# 1. Load and clean dataset
df = pd.read_csv("/content/drive/MyDrive/emails.csv", low_memory=False)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.rename(columns={'Email Text': 'text', 'Email Type': 'label'}, inplace=True)
df.dropna(subset=['text', 'label'], inplace=True)
df = df[df['label'].isin(['Phishing Email', 'Safe Email'])]

# 2. Preprocessing
punct_table = str.maketrans('', '', string.punctuation)
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess(text):
    text = text.lower()
    text = text.translate(punct_table)
    tokens = re.findall(r'\b[a-z]{2,}\b', text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].astype(str).apply(preprocess)

# 3. Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_text'])

# 4. Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Results storage
results = []

def evaluate_model(name, model, params=None):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n🎯 {name} - Params: {params if params else 'Default'}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:\n", cm)

    results.append({
        "Model": name,
        "Params": str(params),
        "Accuracy": report["accuracy"],
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1": report["weighted avg"]["f1-score"]
    })

# 6. Random Forest
rf_params_list = [{"n_estimators": 10}, {"n_estimators": 20}, {"n_estimators": 30}, {"n_estimators": 40}, {"n_estimators": 50}, {"n_estimators": 100}]
for params in rf_params_list:
    model = RandomForestClassifier(**params, random_state=42)
    evaluate_model("Random Forest", model, params)

# 7. SVM
# Trying different C values with linear kernel
svm_params_list = [{"C": 0.1}, {"C": 0.5}, {"C": 1}, {"C": 5}, {"C": 10}, {"C": 100}]
for params in svm_params_list:
    svm_model = SVC(**params, kernel='linear', random_state=42)
    evaluate_model("SVM", svm_model, params)


# 8. Logistic Regression
logreg_params_list = [{"C": 0.1}, {"C": 0.5}, {"C": 1.0}, {"C": 5.0}, {"C": 10.0}]
for params in logreg_params_list:
    model = LogisticRegression(**params, max_iter=1000)
    evaluate_model("Logistic Regression", model, params)

# 9. XGBoost
xgb_params_list = [{"n_estimators": 100, "max_depth": 3}, {"n_estimators": 200, "max_depth": 5}, {"n_estimators": 300, "max_depth": 7}]
for params in xgb_params_list:
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    evaluate_model("XGBoost", model, params)

# 10. Visualize results
results_df = pd.DataFrame(results)

# Barplot: Accuracy comparison
plt.figure(figsize=(14, 6))
sns.barplot(data=results_df, x='Model', y='Accuracy', hue='Params')
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Barplot: F1 Score comparison
plt.figure(figsize=(14, 6))
sns.barplot(data=results_df, x='Model', y='F1', hue='Params')
plt.title("Model F1 Score Comparison")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Line plots: Accuracy change for each model
models = results_df['Model'].unique()
fig, axes = plt.subplots(len(models), 1, figsize=(10, 18), sharex=False)

for i, model in enumerate(models):
    subset = results_df[results_df['Model'] == model]
    axes[i].plot(subset['Params'], subset['Accuracy'], marker='o', label=model)
    axes[i].set_title(f"{model} - Accuracy vs Parameters")
    axes[i].set_ylabel("Accuracy")
    axes[i].set_xlabel("Parameters")
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(True)
    axes[i].legend()

plt.tight_layout()
plt.show()

# Combined plot: Best accuracy of each model
best_results = results_df.groupby("Model").apply(lambda df: df[df['Accuracy'] == df['Accuracy'].max()]).reset_index(drop=True)

plt.figure(figsize=(10, 6))
sns.lineplot(data=best_results, x='Model', y='Accuracy', marker='o')
plt.title("Best Accuracy of Each Model")
plt.ylabel("Accuracy")
plt.ylim(0.5, 1)  # Set y-axis from 0.5 to 1
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. Display final performance table
summary_table = results_df[["Model", "Params", "Accuracy", "Precision", "Recall", "F1"]]
summary_table = summary_table.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

print("\n📊 Final Model Performance Summary Table:\n")
print(summary_table.to_string(index=False))  # Clean formatted table
