"""
Production-Grade Machine Learning Pipeline for Churn Prediction
Author: Victor
Purpose: Build, evaluate, and deploy multiple ML models with business metrics
Business Impact: Enable data-driven retention strategies with measurable ROI
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Load engineered features
print("="*60)
print("MACHINE LEARNING MODEL DEVELOPMENT")
print("="*60)

df = pd.read_csv('../data/telco_churn_scientific_features.csv')
print(f"\nDataset shape: {df.shape}")

# Prepare data for modeling
# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('customerID')
categorical_cols.remove('Churn')

print("\nEncoding categorical variables...")
label_encoders = {}
df_encoded = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

# Prepare features and target
X = df_encoded.drop(['customerID', 'Churn'], axis=1)
y = (df_encoded['Churn'] == 'Yes').astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Churn rate in training: {y_train.mean():.2%}")
print(f"Churn rate in test: {y_test.mean():.2%}")

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
}

# Train and evaluate models
print("\n" + "="*60)
print("MODEL TRAINING AND EVALUATION")
print("="*60)

results = {}
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use appropriate data
    if name in ['Logistic Regression', 'Neural Network']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_prob,
        'auc': auc_score,
        'f1': f1,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    print(f"AUC: {auc_score:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    if auc_score > best_score:
        best_score = auc_score
        best_model = name

print(f"\nBest Model: {best_model} (AUC: {best_score:.4f})")

# Business Metrics Calculation
print("\n" + "="*60)
print("BUSINESS IMPACT ANALYSIS")
print("="*60)

# Cost-benefit analysis
avg_customer_value = df['MonthlyCharges'].mean() * 12  # Annual value
retention_cost = 50  # Assumed cost per retention campaign
acquisition_cost = 500  # Industry average

for name, result in results.items():
    cm = confusion_matrix(y_test, result['predictions'])
    tn, fp, fn, tp = cm.ravel()
    
    # Business metrics
    prevented_churn_value = tp * avg_customer_value
    false_alarm_cost = fp * retention_cost
    missed_churn_cost = fn * acquisition_cost
    
    net_benefit = prevented_churn_value - false_alarm_cost - missed_churn_cost
    roi = (net_benefit / (false_alarm_cost + retention_cost * tp)) * 100
    
    result['business_metrics'] = {
        'prevented_churn_value': prevented_churn_value,
        'false_alarm_cost': false_alarm_cost,
        'missed_churn_cost': missed_churn_cost,
        'net_benefit': net_benefit,
        'roi': roi
    }
    
    print(f"\n{name}:")
    print(f"  Prevented Churn Value: ${prevented_churn_value:,.0f}")
    print(f"  False Alarm Cost: ${false_alarm_cost:,.0f}")
    print(f"  Missed Churn Cost: ${missed_churn_cost:,.0f}")
    print(f"  Net Benefit: ${net_benefit:,.0f}")
    print(f"  ROI: {roi:.1f}%")

# Create comprehensive visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Model Performance Comparison', 'ROC Curves',
                    'Business Impact by Model', 'Feature Importance'),
    specs=[[{'type': 'bar'}, {'type': 'scatter'}],
           [{'type': 'bar'}, {'type': 'bar'}]]
)

# 1. Model Performance Comparison
model_names = list(results.keys())
aucs = [results[m]['auc'] for m in model_names]
f1s = [results[m]['f1'] for m in model_names]

fig.add_trace(
    go.Bar(name='AUC', x=model_names, y=aucs, 
           text=[f'{v:.3f}' for v in aucs], textposition='auto'),
    row=1, col=1
)
fig.add_trace(
    go.Bar(name='F1', x=model_names, y=f1s,
           text=[f'{v:.3f}' for v in f1s], textposition='auto'),
    row=1, col=1
)

# 2. ROC Curves
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={result['auc']:.3f})",
                   mode='lines'),
        row=1, col=2
    )
fig.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], name='Random', 
               line=dict(dash='dash', color='gray')),
    row=1, col=2
)

# 3. Business Impact
net_benefits = [results[m]['business_metrics']['net_benefit'] for m in model_names]
rois = [results[m]['business_metrics']['roi'] for m in model_names]

fig.add_trace(
    go.Bar(name='Net Benefit ($)', x=model_names, y=net_benefits,
           text=[f'${v:,.0f}' for v in net_benefits], textposition='auto'),
    row=2, col=1
)

# 4. Feature Importance (using best model)
if best_model in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    importances = results[best_model]['model'].feature_importances_
    indices = np.argsort(importances)[-10:]
    top_features = X.columns[indices]
    top_importances = importances[indices]
    
    fig.add_trace(
        go.Bar(x=top_importances, y=top_features, orientation='h',
               text=[f'{v:.3f}' for v in top_importances], textposition='auto'),
        row=2, col=2
    )

fig.update_layout(height=1000, showlegend=True, 
                  title_text="<b>Machine Learning Model Results</b>",
                  title_font_size=24)
fig.write_html('../outputs/visualizations/model_results.html')

# SHAP Analysis for interpretability
print("\n" + "="*60)
print("MODEL INTERPRETABILITY (SHAP)")
print("="*60)

# Use XGBoost for SHAP analysis
best_xgb = results['XGBoost']['model']
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

# Create SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False, max_display=15)
plt.title('SHAP Feature Importance - What Drives Churn?')
plt.tight_layout()
plt.savefig('../outputs/visualizations/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# Save models and preprocessors
print("\n" + "="*60)
print("SAVING MODELS AND ARTIFACTS")
print("="*60)

# Save best model
joblib.dump(results[best_model]['model'], f'../models/best_model_{best_model.replace(" ", "_")}.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
joblib.dump(label_encoders, '../models/label_encoders.pkl')

# Save model comparison
model_comparison = pd.DataFrame({
    'Model': model_names,
    'AUC': aucs,
    'F1_Score': f1s,
    'Net_Benefit': net_benefits,
    'ROI_Percentage': rois
})
model_comparison.to_csv('../outputs/model_comparison.csv', index=False)

# Create deployment ready prediction function
def predict_churn(customer_data):
    """Production-ready prediction function"""
    # Encode categorical variables
    for col, le in label_encoders.items():
        if col in customer_data:
            customer_data[col] = le.transform([customer_data[col]])[0]
    
    # Scale if needed
    if best_model in ['Logistic Regression', 'Neural Network']:
        features = scaler.transform([customer_data])
    else:
        features = [customer_data]
    
    # Predict
    probability = results[best_model]['model'].predict_proba(features)[0, 1]
    risk_level = 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
    
    return {
        'churn_probability': probability,
        'risk_level': risk_level,
        'recommended_action': get_recommendation(probability, customer_data)
    }

def get_recommendation(probability, customer_data):
    """Generate personalized retention recommendation"""
    if probability > 0.7:
        return "Immediate intervention required: Offer contract upgrade with 20% discount"
    elif probability > 0.5:
        return "Proactive engagement needed: Bundle services with loyalty rewards"
    elif probability > 0.3:
        return "Monitor closely: Send satisfaction survey and personalized offers"
    else:
        return "Low risk: Focus on upselling additional services"

print("\nModel artifacts saved:")
print("- Best model pickle file")
print("- Scaler and encoders")
print("- Model comparison CSV")
print("- SHAP interpretability plots")
print("\nNext steps: Run 04_business_impact_dashboard.py")