import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import KFold # 10-Fold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

df = pd.read_csv("/home/vent/vscode/vscode./ML/project/formResponse.csv")

COL_MAPPING = {
    'Year of study:': 'Year',
    'Living situation': 'Living',
    'I feel that I lack companionship.': 'Lack_Comp',
    'I feel left out.': 'Left_Out',
    'I feel isolated from others.': 'Isolated',
    'I have friends I can talk about important things with.': 'Talk_Friends',
    'I feel understood by people close to me.': 'Understood',
    'I have enough people I can rely on.': 'Rely_On',
    'I often feel people don’t know the real me.': 'Unknown_Me',
    'My relationships feel meaningful to me.': 'Meaningful_Rels',
    'How often do you avoid social activities because of money?': 'Avoid_Money',
    'How many times per week do you usually meet or hang out with friends in person?': 'In_Person_Meetings',
    'On average, how many hours per week do you spend outside your home or dorm (cafes, parks, social events, etc.)?': 'Time_Outside_Home',
    'How often do you participate in social events (such as parties, clubs, or group activities) on weekends?': 'Weekend_Social',
    'On a typical day, how many hours do you spend on your phone (not for study)?': 'Phone_Hours',
    'How often do you check your phone late at night (after midnight)?': 'Phone_Late_Night',
    'Typical nightly sleep': 'Sleep_Hours',
    'Do you feel your sleep is restful?': 'Sleep_Restful',
    'How many days per week do you do moderate or vigorous physical activity?': 'Physical_Activity',
    'Over the last two weeks, how often have you felt down, depressed, or hopeless?': 'Depressed',
    'Over the last two weeks, how often have you had little interest or pleasure in doing things?': 'Little_Interest',
    'I feel comfortable asking classmates for help.': 'Ask_Help',
    'What helps you feel less lonely? (one sentence)': 'Less_Lonely_Text'
}
df.rename(columns=COL_MAPPING, inplace=True)

# REVERSE SCORING
# Meaningful_Rels is positive (5=Good). We reverse it so 5=Bad (High Loneliness).
df['Meaningful_Rels_R'] = 6 - df['Meaningful_Rels']
df.drop('Meaningful_Rels', axis=1, inplace=True)


# TARGET (Y) CREATION
# Use 5 core questions to create a custom loneliness score (Max score = 5 * 5 = 25)
target_questions = ['Lack_Comp', 'Left_Out', 'Isolated', 'Unknown_Me', 'Meaningful_Rels_R']

df['Custom_Loneliness_Score'] = df[target_questions].sum(axis=1)

# CLASSIFICATION TARGET: If score is above 15, classify as High Loneliness (1)
CUTOFF = 15
df['Loneliness_Level'] = np.where(df['Custom_Loneliness_Score'] > CUTOFF, 1, 0)

print(f"Target distribution (0=Low, 1=High Loneliness, using cutoff > {CUTOFF}):")
print(df['Loneliness_Level'].value_counts())
print("\n")

# FINAL FEATURE SELECTION (X)
X = df.drop(columns=['Timestamp', 'Custom_Loneliness_Score', 'Less_Lonely_Text', 'Loneliness_Level'])
Y = df['Loneliness_Level']

# Define feature groups for preprocessing
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist() # Should be just 'Living'

# Remove features that are implicitly handled or removed later
if 'Year' in numerical_features: numerical_features.remove('Year') # Treat Year as ordinal/categorical, but use numerical values
if 'Living' in categorical_features: categorical_features.remove('Living')

# Re-confirm the final lists
numerical_features = [
    'Lack_Comp', 'Left_Out', 'Isolated', 'Talk_Friends', 'Understood', 
    'Rely_On', 'Unknown_Me', 'Ask_Help', 'Meaningful_Rels_R', 'Avoid_Money', 
    'In_Person_Meetings', 'Time_Outside_Home', 'Weekend_Social',
    'Phone_Hours', 'Phone_Late_Night', 'Sleep_Hours', 'Sleep_Restful',
    'Physical_Activity', 'Depressed', 'Little_Interest', 'Year' 
]
categorical_features = ['Living']


# This handles scaling for numerical and one-hot encoding for categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store results
accuracy_scores = []
f1_scores = []
coefficients_list = []
fold_number = 1


print("--- Starting 10-Fold Cross-Validation (K=10) ---")

# We use the full X and Y since KFold handles the splitting internally
for train_index, test_index in kf.split(X, Y):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    # Preprocess Data: Fit Scaler/Encoder on Training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train Logistic Regression Model
    model = LogisticRegression(random_state=42, solver='liblinear') 
    model.fit(X_train_processed, Y_train)
    
    # Evaluate
    Y_pred = model.predict(X_test_processed)
    
    # Store key metrics
    accuracy_scores.append(accuracy_score(Y_test, Y_pred))
    f1_scores.append(f1_score(Y_test, Y_pred, average='weighted', zero_division=0))
    coefficients_list.append(model.coef_[0])
    
    print(f"Fold {fold_number} | Accuracy: {accuracy_scores[-1]:.4f} | F1-Score: {f1_scores[-1]:.4f}")
    fold_number += 1


    # Calculate and Display Average Performance
print("\n--- Summary of 10-Fold Cross-Validation ---")
print(f"Average Accuracy across 10 folds: {np.mean(accuracy_scores):.4f}")
print(f"Average Weighted F1-Score: {np.mean(f1_scores):.4f}")


# We use the coefficients from the last fold for visualization, as it represents a fully trained model.

# 1-Get the final feature names after One-Hot Encoding
numeric_names = preprocessor.named_transformers_['num'].get_feature_names_out(numerical_features)
categorical_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
feature_names = np.concatenate([numeric_names, categorical_names])

# 2. Extract coefficients from the last model trained
coefficients_to_plot = coefficients_list[-1]

# 3.Create a DataFrame for sorting and plotting
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients_to_plot
})

# Sort by the magnitude (absolute value) to find the most influential features
feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)

print("\n--- Top 10 Most Influential Predictors ---")
print("Positive Coefficient (Red) -> Increases likelihood of HIGH Loneliness")
print("Negative Coefficient (Blue) -> Decreases likelihood of HIGH Loneliness\n")
print(feature_importance[['Feature', 'Coefficient']].head(10).round(3))


# 4. Visualization (Matplotlib)
top_n = 10
plot_data = feature_importance.head(top_n).sort_values(by='Coefficient', ascending=True)

plt.figure(figsize=(10, 6))

colors = ['red' if c > 0 else 'blue' for c in plot_data['Coefficient']]

plt.barh(plot_data['Feature'], plot_data['Coefficient'], color=colors)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)

plt.xlabel('Coefficient Value (Predictive Power on Loneliness)')
plt.title(f'Top {top_n} Predictors of High Loneliness (Logistic Regression)')
plt.suptitle('Blue = Predicts Low Loneliness | Red = Predicts High Loneliness', fontsize=10)
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# plt.show()

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X, Y)



joblib.dump(pipeline, "/home/vent/vscode/vscode./ML/project/loneliness_model.pkl")
