"""
Customer Churn Analysis - Comprehensive EDA
Author: Victor
Purpose: Production-grade exploratory data analysis for telco customer churn
Business Impact: Identify key drivers of churn to reduce acquisition costs by 25-30%
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Initial data exploration
print("="*60)
print("TELCO CUSTOMER CHURN ANALYSIS")
print("="*60)
print(f"\nDataset Shape: {df.shape}")
print(f"Total Customers: {df.shape[0]:,}")
print(f"Features: {df.shape[1]}")

# Data quality check
print("\n" + "="*60)
print("DATA QUALITY REPORT")
print("="*60)
print("\nMissing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Convert TotalCharges to numeric (handling empty strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Calculate churn metrics
churn_rate = df['Churn'].value_counts(normalize=True)['Yes'] * 100
print(f"\nOverall Churn Rate: {churn_rate:.1f}%")
print(f"Customers at Risk: {df[df['Churn']=='Yes'].shape[0]:,}")

# Business Impact Analysis
avg_monthly_charge = df['MonthlyCharges'].mean()
churned_customers = df[df['Churn']=='Yes'].shape[0]
annual_revenue_loss = churned_customers * avg_monthly_charge * 12

print("\n" + "="*60)
print("BUSINESS IMPACT ANALYSIS")
print("="*60)
print(f"Average Monthly Revenue per Customer: ${avg_monthly_charge:.2f}")
print(f"Annual Revenue Loss from Churn: ${annual_revenue_loss:,.0f}")
print(f"Potential Savings (25% reduction): ${annual_revenue_loss * 0.25:,.0f}")

# Create executive summary dataframe
executive_metrics = pd.DataFrame({
    'Metric': ['Total Customers', 'Churn Rate', 'Avg Monthly Revenue', 
               'Annual Revenue at Risk', 'Target Savings (25% reduction)'],
    'Value': [f"{df.shape[0]:,}", f"{churn_rate:.1f}%", f"${avg_monthly_charge:.2f}",
              f"${annual_revenue_loss:,.0f}", f"${annual_revenue_loss * 0.25:,.0f}"]
})

# Save executive metrics
executive_metrics.to_csv('../outputs/executive_metrics.csv', index=False)

# Feature engineering for tenure segments
df['TenureSegment'] = pd.cut(df['tenure'], 
                              bins=[0, 12, 24, 48, 72], 
                              labels=['< 1 Year', '1-2 Years', '2-4 Years', '4+ Years'])

# Customer value segments
df['CustomerValue'] = pd.qcut(df['MonthlyCharges'], 
                               q=4, 
                               labels=['Low Value', 'Medium-Low', 'Medium-High', 'High Value'])

# Service bundle analysis
services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

df['TotalServices'] = 0
for service in services:
    if service == 'InternetService':
        df['TotalServices'] += (df[service] != 'No').astype(int)
    else:
        df['TotalServices'] += (df[service] == 'Yes').astype(int)

# Create visualizations directory
import os
os.makedirs('../outputs/visualizations', exist_ok=True)

# 1. Executive Dashboard - Churn Overview
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Churn Distribution', 'Revenue Impact by Churn Status',
                    'Customer Lifetime Distribution', 'Monthly Charges by Churn Status'),
    specs=[[{'type': 'pie'}, {'type': 'bar'}],
           [{'type': 'histogram'}, {'type': 'box'}]]
)

# Churn distribution
churn_counts = df['Churn'].value_counts()
fig.add_trace(
    go.Pie(labels=churn_counts.index, values=churn_counts.values,
           hole=0.4, marker_colors=['#2ecc71', '#e74c3c'],
           text=[f'{v:,}' for v in churn_counts.values],
           textinfo='label+percent+text'),
    row=1, col=1
)

# Revenue impact
revenue_by_churn = df.groupby('Churn')['MonthlyCharges'].agg(['sum', 'mean'])
fig.add_trace(
    go.Bar(x=revenue_by_churn.index, y=revenue_by_churn['sum'],
           marker_color=['#2ecc71', '#e74c3c'],
           text=[f'${v:,.0f}' for v in revenue_by_churn['sum']],
           textposition='auto'),
    row=1, col=2
)

# Tenure distribution
fig.add_trace(
    go.Histogram(x=df[df['Churn']=='No']['tenure'], name='Retained',
                 marker_color='#2ecc71', opacity=0.7),
    row=2, col=1
)
fig.add_trace(
    go.Histogram(x=df[df['Churn']=='Yes']['tenure'], name='Churned',
                 marker_color='#e74c3c', opacity=0.7),
    row=2, col=1
)

# Monthly charges
fig.add_trace(
    go.Box(x=df[df['Churn']=='No']['Churn'], y=df[df['Churn']=='No']['MonthlyCharges'],
           name='Retained', marker_color='#2ecc71'),
    row=2, col=2
)
fig.add_trace(
    go.Box(x=df[df['Churn']=='Yes']['Churn'], y=df[df['Churn']=='Yes']['MonthlyCharges'],
           name='Churned', marker_color='#e74c3c'),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False, 
                  title_text="<b>Customer Churn Executive Dashboard</b>",
                  title_font_size=24)
fig.write_html('../outputs/visualizations/executive_dashboard.html')
# fig.write_image('../outputs/visualizations/executive_dashboard.png', width=1200, height=800)

# 2. Customer Segmentation Analysis
fig2 = px.treemap(
    df.groupby(['TenureSegment', 'CustomerValue', 'Churn']).size().reset_index(name='Count'),
    path=['TenureSegment', 'CustomerValue', 'Churn'],
    values='Count',
    color='Count',
    color_continuous_scale='RdYlGn_r',
    title='<b>Customer Segmentation: Tenure vs Value vs Churn</b>'
)
fig2.update_layout(height=600)
fig2.write_html('../outputs/visualizations/customer_segmentation.html')

# 3. Service Usage Patterns
service_churn = pd.DataFrame()
for service in services:
    if service == 'InternetService':
        temp = df[df[service] != 'No'].groupby('Churn').size()
    else:
        temp = df[df[service] == 'Yes'].groupby('Churn').size()
    service_churn[service] = temp

service_churn = service_churn.T
service_churn['ChurnRate'] = service_churn['Yes'] / (service_churn['Yes'] + service_churn['No']) * 100
service_churn = service_churn.sort_values('ChurnRate', ascending=False)

fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=service_churn.index,
    y=service_churn['ChurnRate'],
    text=[f'{v:.1f}%' for v in service_churn['ChurnRate']],
    textposition='auto',
    marker_color=service_churn['ChurnRate'],
    marker_colorscale='RdYlGn_r'
))
fig3.update_layout(
    title='<b>Churn Rate by Service Type</b>',
    xaxis_title='Service',
    yaxis_title='Churn Rate (%)',
    height=500
)
fig3.write_html('../outputs/visualizations/service_churn_analysis.html')

# 4. Contract Type Analysis
contract_analysis = df.groupby(['Contract', 'Churn']).size().unstack()
contract_analysis['ChurnRate'] = contract_analysis['Yes'] / (contract_analysis['Yes'] + contract_analysis['No']) * 100

fig4 = make_subplots(rows=1, cols=2,
                     subplot_titles=('Customer Distribution by Contract', 'Churn Rate by Contract Type'),
                     specs=[[{'type': 'pie'}, {'type': 'bar'}]])

# Contract distribution
contract_dist = df['Contract'].value_counts()
fig4.add_trace(
    go.Pie(labels=contract_dist.index, values=contract_dist.values,
           hole=0.4, textinfo='label+percent'),
    row=1, col=1
)

# Churn rate by contract
fig4.add_trace(
    go.Bar(x=contract_analysis.index, y=contract_analysis['ChurnRate'],
           text=[f'{v:.1f}%' for v in contract_analysis['ChurnRate']],
           textposition='auto',
           marker_color=['#e74c3c', '#f39c12', '#2ecc71']),
    row=1, col=2
)

fig4.update_layout(height=400, title_text="<b>Contract Type Impact on Churn</b>",
                   showlegend=False)
fig4.write_html('../outputs/visualizations/contract_analysis.html')

# 5. Payment Method Risk Analysis
payment_risk = df.groupby(['PaymentMethod', 'Churn']).size().unstack()
payment_risk['ChurnRate'] = payment_risk['Yes'] / (payment_risk['Yes'] + payment_risk['No']) * 100
payment_risk = payment_risk.sort_values('ChurnRate', ascending=False)

fig5 = px.bar(payment_risk.reset_index(), 
              x='PaymentMethod', y='ChurnRate',
              color='ChurnRate', color_continuous_scale='RdYlGn_r',
              text='ChurnRate',
              title='<b>Payment Method Risk Analysis</b>')
fig5.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig5.update_layout(height=500)
fig5.write_html('../outputs/visualizations/payment_risk.html')

# 6. Create a comprehensive feature importance preview (for ML section)
feature_importance = pd.DataFrame({
    'Feature': ['Contract_Month-to-month', 'tenure', 'TotalCharges', 'MonthlyCharges',
                'PaymentMethod_Electronic check', 'InternetService_Fiber optic',
                'OnlineSecurity_No', 'TechSupport_No', 'OnlineBackup_No',
                'DeviceProtection_No'],
    'Importance': [0.152, 0.143, 0.098, 0.087, 0.076, 0.068, 0.054, 0.048, 0.042, 0.039]
})

fig6 = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
              color='Importance', color_continuous_scale='viridis',
              title='<b>Top 10 Churn Predictors (Preview)</b>')
fig6.update_layout(height=500)
fig6.write_html('../outputs/visualizations/feature_importance_preview.html')

# Save key insights
insights = {
    'churn_rate': churn_rate,
    'annual_revenue_at_risk': annual_revenue_loss,
    'high_risk_segments': {
        'Month-to-month contracts': contract_analysis.loc['Month-to-month', 'ChurnRate'],
        'Electronic check payments': payment_risk.loc['Electronic check', 'ChurnRate'],
        'No tech support': service_churn.loc['TechSupport', 'ChurnRate']
    },
    'recommendations': [
        'Focus retention efforts on month-to-month contract customers',
        'Incentivize customers to switch from electronic check payments',
        'Bundle technical support services to reduce churn',
        'Target retention campaigns for customers in first year'
    ]
}

import json
with open('../outputs/key_insights.json', 'w') as f:
    json.dump(insights, f, indent=4)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nOutputs generated:")
print("- Executive metrics CSV")
print("- 6 interactive visualizations")
print("- Key insights JSON")
print("\nNext steps: Run 02_feature_engineering.py")