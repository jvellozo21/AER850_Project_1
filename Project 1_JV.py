#AER 850 Section 2
#Jayden Vellozo
#501106139
#Project 1

#STEP 1 - Data Processing
import numpy as np
import pandas as pd

data = pd.read_csv("Project 1 Data.csv")

print(data.head())  #looks at first few rows 
print(data.shape) #rows and columns

#STEP 2 DATA VISUALIZATION 
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

print("=== BASIC INFO ===")
print(data.info()) #info on data types and non-null counts
print("\n=== BASIC STATISTICS ===")
print(data.describe()) #summary statistics for numerical columns

unique_steps = data['Step'].unique()  #checks unique values in Step column
print("\nUnique Step classes:", unique_steps)
print("Number of classes:", len(unique_steps))

#Plot Histogrms
data.hist(figsize=(10, 6), bins=20, edgecolor="black")
plt.suptitle("Histogram for Each Feature")
plt.savefig(Path("histograms.png")), plt.close()

#  Scatter plots to visualize relationships between variables
#  These help us see how X, Y, and Z vary with each other
plt.figure(figsize=(10, 5))
plt.scatter(data["X"], data["Z"], c=data["Step"], cmap="viridis")
plt.title("X vs Z colored by Step class")
plt.xlabel("X"), plt.ylabel("Z")
plt.colorbar(label="Step Class")
plt.savefig(Path("scatter_XZ.png")),plt.close()

plt.figure(figsize=(10, 5))
plt.scatter(data["Y"], data["Z"], c=data["Step"], cmap="plasma")
plt.title("Y vs Z colored by Step class")
plt.xlabel("Y"), plt.ylabel("Z")
plt.colorbar(label="Step Class")
plt.savefig(Path("scatter_YZ.png")), plt.close() 

#Boxplots show the median, spread, and outliers for each class
plt.figure(figsize=(10, 5))
sns.boxplot(x="Step", y="X", data=data)
plt.title("Boxplot of X by Step class")
plt.savefig(Path("boxplot_X.png")), plt.close() 

plt.figure(figsize=(10, 5))
sns.boxplot(x="Step", y="Y", data=data)
plt.title("Boxplot of Y by Step class")
plt.savefig(Path("boxplot_Y.png")),plt.close()

plt.figure(figsize=(10, 5))
sns.boxplot(x="Step", y="Z", data=data)
plt.title("Boxplot of Z by Step class")
plt.savefig(Path("boxplot_Z.png")), plt.close()

#STEP 3 CORRELATION ANALYSIS
# Computing the Pearson Correlation which measures the linear relationship between variabes
# Values range from -1 to +1 
#   +1: perfect positive correlation (as one increases, the other increases)
#   -1: perfect negative correlation (as one increases, the other decreases)
#    0: no linear correlation
corr_matrix = data.corr(method='pearson') 
print("=== CORRELATTION MATRIX (Pearson) ===")
print(corr_matrix)

#heatmap for better visualization
plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)      
plt.title("Pearson Correlation Heatmap")
plt.savefig(Path("correlation_heatmap.png")),plt.close()

# STEP 4 - MODEL DEVELOPMENT & HYPERPARAMETER TUNING
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Split features/labels + train/test (stratified to keep class balance)
X = data[['X', 'Y', 'Z']].values
y = data['Step'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

#Cross-validation strategy (5-fold, stratified, shuffled)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Three GridSearchCV models: Logistic Regression, Decision Tree and Random Forest
#Logistic Regression (GridSearchCV)
pipe_lr = Pipeline([
    ('scaler', StandardScaler()), ('clf', LogisticRegression( solver='lbfgs', max_iter=1000, random_state=42))
])
param_lr = {'clf__C': [0.1, 1, 10, 100]}
grid_lr = GridSearchCV(pipe_lr, param_lr, cv=cv, scoring='f1_macro', n_jobs=-1) 
grid_lr.fit(X_train, y_train)

#Random Forest (GridSearchCV)
rf = RandomForestClassifier(random_state=42)
param_rf = {
    'n_estimators': [50, 100, 200, 300], #number of trees in the forest
    'max_depth': [None, 10, 20, 30, 40], #maximum depth of the tree None means nodes are expanded until all leaves are pure
    'min_samples_split': [2, 5, 10], #minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4], #minimum number of samples required to be at a leaf node
    'max_features': ['sqrt'], #number of features to consider when looking for the best split (sqrt = square root of total features)
    'criterion': ['gini'] #function to measure the quality of a split
}
grid_rf = GridSearchCV(
    rf, param_rf, cv=cv, scoring='f1_macro', n_jobs=-1
)
grid_rf.fit(X_train, y_train)

#Decision Tree (GridSearchCV)
dt = DecisionTreeClassifier(random_state=42)
param_dt = {
    'max_depth': [None, 5, 10, 20, 30], #maximum depth of the tree
    'min_samples_split': [10, 20], 
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'] #function to measure the quality of a split 
}
grid_dt = GridSearchCV(dt, param_dt, cv=cv, scoring='f1_macro', n_jobs=-1)
grid_dt.fit(X_train, y_train)

#SVM (RandomizedSearchCV)
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC())
])
param_svm = {
    'clf__kernel': ['rbf', 'linear', 'poly'], #radial basis function kernel
    'clf__C': [0.1, 1, 10, 100, 1000], #regularization parameter
    'clf__gamma': ['scale', 0.1, 0.01, 0.001]
}
rand_svm = RandomizedSearchCV(
    pipe_svm, param_svm, n_iter=10, cv=cv, scoring='f1_macro', random_state=42, n_jobs=-1
)
rand_svm.fit(X_train, y_train)

#STEP 5 - FINAL EVALUATION ON TEST SET
#Collect best estimators and cross-validated macro-F1 scores
best_models = {
    'LogisticRegression': (grid_lr.best_estimator_, grid_lr.best_score_, grid_lr.best_params_),
    'DecisionTree':       (grid_dt.best_estimator_, grid_dt.best_score_, grid_dt.best_params_),
    'RandomForest':       (grid_rf.best_estimator_, grid_rf.best_score_, grid_rf.best_params_),
    'SVM':                (rand_svm.best_estimator_, rand_svm.best_score_, rand_svm.best_params_)
}

summary_rows = []
for name, (est, score, params) in best_models.items():
    summary_rows.append({'Model': name, 'CV_F1_macro': round(score, 4)})
    print(f"\n{name}:")
    print(f"CV F1 Score (macro): {score:.4f}")
    print("Best Parameters:")
    for param, value in params.items():
        print(f"  {param}: {value}")

summary_df = pd.DataFrame(summary_rows).sort_values('CV_F1_macro', ascending=False)

#Store best models for next steps
best_lr  = best_models['LogisticRegression'][0]
best_svm = best_models['SVM'][0]
best_dt  = best_models['DecisionTree'][0]
best_rf  = best_models['RandomForest'][0]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def eval_test(name, est):
    y_pred = est.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    p    = precision_score(y_test, y_pred, average='macro', zero_division=0)
    r    = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1m  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"{name:16s} | Acc: {acc:.3f}  Prec(m): {p:.3f}  Rec(m): {r:.3f}  F1(m): {f1m:.3f}")
    return y_pred, f1m

print("\n=== TEST SET RESULTS ===")
preds = {}
preds['LR'],  f1_lr  = eval_test("LogisticRegression", best_models['LogisticRegression'][0])
preds['SVM'], f1_svm = eval_test("SVM",          best_models['SVM'][0])
preds['DT'],  f1_dt  = eval_test("DecisionTree",     best_models['DecisionTree'][0])
preds['RF'],  f1_rf  = eval_test("RandomForest",     best_models['RandomForest'][0])

# === TEST-SET MODEL COMPARISON CHART ===
test_rows = [
    {"Model": "LogisticRegression", "F1_macro": f1_lr},
    {"Model": "SVM",                "F1_macro": f1_svm},
    {"Model": "DecisionTree",       "F1_macro": f1_dt},
    {"Model": "RandomForest",       "F1_macro": f1_rf},
]
test_df = pd.DataFrame(test_rows).sort_values("F1_macro", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(test_df["Model"], test_df["F1_macro"], color="skyblue", edgecolor="black")
plt.ylabel("Test F1 (macro)")
plt.title("Model Performance Comparison (Test Set)")
plt.xticks(rotation=45)
for i, v in enumerate(test_df["F1_macro"]):
    plt.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("model_comparison.png"), plt.close()

# pick winner by F1-macro
best_name = max([('LR', f1_lr), ('SVM', f1_svm), ('DT', f1_dt), ('RF', f1_rf)], key=lambda x: x[1])[0]
best_label = {'LR':'LogisticRegression','SVM':'SVM','DT':'DecisionTree','RF':'RandomForest'}[best_name]
best_est   = best_models[best_label][0]
y_pred     = preds[best_name]

print(f"\nBest on TEST by F1(macro): {best_label}")

# confusion matrix
labels = np.unique(y_test)
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion Matrix — {best_label} (counts)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.savefig('confusion_matrix.png'), plt.close()

#Step 6 -  Stacked Model Performance Analysis
from sklearn.ensemble import StackingClassifier

#Top 2 models based on test F1-macro
ranked = summary_df[["Model", "CV_F1_macro"]].values.tolist() 
ranked.sort(key=lambda x: x[1], reverse=True) 
top1, top2 = ranked[0][0], ranked[1][0]

base1 = best_models[top1][0]   # best_estimator_ for model 1
base2 = best_models[top2][0]   # best_estimator_ for model 2

print(f"\nStacking base models: {top1} + {top2}")

# 2) Build the stack (meta-learner = Logistic Regression)
stack = StackingClassifier(
    estimators=[(top1, base1), (top2, base2)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    passthrough=False,
    n_jobs=-1
)

# 3) Fit on train and evaluate on test
stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)

acc  = accuracy_score(y_test, y_pred_stack)
prec = precision_score(y_test, y_pred_stack, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred_stack, average="macro", zero_division=0)
f1m  = f1_score(y_test, y_pred_stack, average="macro", zero_division=0)

print(f"\nStacked ({top1} + {top2}) — Test Acc: {acc:.3f} | Prec(m): {prec:.3f} | Rec(m): {rec:.3f} | F1(m): {f1m:.3f}")

# 4) Confusion matrix for the stack
labels = np.unique(y_test)
cm = confusion_matrix(y_test, y_pred_stack, labels=labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion Matrix — Stacked ({top1} + {top2})")
plt.xlabel("Predicted"); 
plt.ylabel("True")
plt.tight_layout(); 
plt.savefig('stacked_confusion_matrix.png'); plt.close()

# Step 7 - MODEL Evaluation on New Data
from joblib import dump, load
final_model = stack
#Save the stacked model to a joblib file
dump(final_model, 'best_model.joblib')
print("Stacked model saved successfully as 'best_model.joblib'.")

#Load the saved stacked model
loaded_model = load('best_model.joblib')
print("Stacked model loaded successfully.")

#Define new coordinate data for prediction
new_coordinates = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0.0, 3.0625, 1.93],
    [9.4, 3.0,    1.8],
    [9.4, 3.0,    1.3]
])
#Predict the corresponding maintenance step
predicted_steps = loaded_model.predict(new_coordinates)
#Display predictions
for i, coords in enumerate(new_coordinates):
    print(f"Coordinates {coords}  Predicted Step: {predicted_steps[i]}")