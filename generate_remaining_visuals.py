"""
Generate Remaining High-Quality Visualizations
Author: Victor
Purpose: Create LinkedIn and GitHub showcase images
"""

import matplotlib.pyplot as plt
import numpy as np

# Set high quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("Generating remaining visualizations...")

# 7. LINKEDIN SHOWCASE IMAGE
print("Creating LinkedIn Showcase Image...")

fig_linkedin = plt.figure(figsize=(12, 8))
fig_linkedin.patch.set_facecolor('#0077B5')  # LinkedIn blue

# Main title
fig_linkedin.text(0.5, 0.92, 'AI-POWERED CUSTOMER CHURN PREDICTION', 
                  ha='center', fontsize=26, fontweight='bold', color='white')
fig_linkedin.text(0.5, 0.86, 'Reducing Telco Customer Attrition by 25%', 
                  ha='center', fontsize=18, color='white')

# Key achievements
achievements = """
26.5% → 19.9% Churn Rate Reduction
$363K Annual Revenue Saved
430% ROI on Retention Campaigns
84.9% Model Accuracy (AUC)
5-Month Payback Period
"""

fig_linkedin.text(0.25, 0.45, achievements, fontsize=20, va='center', color='white',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='#0066CC', alpha=0.8))

# Tech stack
tech_stack = """
Tech Stack:
• Python (Pandas, Scikit-learn, XGBoost)
• Advanced Feature Engineering
• Ensemble Machine Learning
• SHAP for Model Interpretability
• Streamlit Interactive Dashboard
"""

fig_linkedin.text(0.75, 0.45, tech_stack, fontsize=14, va='center', color='white',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='#004182', alpha=0.8))

# Call to action
fig_linkedin.text(0.5, 0.08, 'View Full Project: github.com/yourusername | Connect on LinkedIn', 
                  ha='center', fontsize=12, color='white', style='italic')

plt.axis('off')
plt.tight_layout()
plt.savefig('../outputs/visualizations/linkedin_showcase.png', bbox_inches='tight', facecolor='#0077B5')
plt.savefig('../outputs/visualizations/linkedin_showcase.jpg', bbox_inches='tight', facecolor='#0077B5')
plt.close()

# 8. GITHUB README HEADER
print("Creating GitHub README Header...")

fig_github = plt.figure(figsize=(16, 6))
fig_github.patch.set_facecolor('#24292e')  # GitHub dark

# Project title
fig_github.text(0.5, 0.75, 'Customer Churn Prediction & Retention Strategy', 
                ha='center', fontsize=32, fontweight='bold', color='white')

# Badges simulation
badges_text = "ML Project  |  Business Impact  |  84.9% Accuracy  |  430% ROI"
fig_github.text(0.5, 0.45, badges_text, ha='center', fontsize=18, color='#58a6ff')

# Brief description
desc_text = "Production-ready machine learning solution for predicting customer churn with actionable business insights"
fig_github.text(0.5, 0.20, desc_text, ha='center', fontsize=16, color='#8b949e', style='italic')

plt.axis('off')
plt.tight_layout()
plt.savefig('../outputs/visualizations/github_header.png', bbox_inches='tight', facecolor='#24292e')
plt.savefig('../outputs/visualizations/github_header.jpg', bbox_inches='tight', facecolor='#24292e')
plt.close()

# 9. PORTFOLIO HERO IMAGE
print("Creating Portfolio Hero Image...")

fig_hero = plt.figure(figsize=(16, 9))
ax = fig_hero.add_subplot(111)

# Create gradient background
gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient = np.vstack((gradient, gradient))
ax.imshow(gradient, extent=[0, 10, 0, 10], aspect='auto', cmap='Blues_r', alpha=0.8)

# Add title and metrics
ax.text(5, 8, 'CUSTOMER CHURN PREDICTION PROJECT', 
        ha='center', fontsize=32, fontweight='bold', color='darkblue',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))

metrics_text = """
Business Impact
• 26.5% Churn Reduction
• $363K Revenue Saved
• 430% ROI

Technical Excellence
• 84.9% Model Accuracy
• 6 ML Algorithms
• 42 Engineered Features

Deliverables
• Predictive Model
• Interactive Dashboard
• Strategic Recommendations
"""

ax.text(2.5, 4, metrics_text, fontsize=16, va='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))

# Add visual elements
ax.text(7.5, 2, '🚀', fontsize=80, ha='center', alpha=0.3)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
plt.tight_layout()
plt.savefig('../outputs/visualizations/portfolio_hero.png', bbox_inches='tight', facecolor='white')
plt.savefig('../outputs/visualizations/portfolio_hero.jpg', bbox_inches='tight', facecolor='white')
plt.close()

# 10. RESULTS COMPARISON CHART
print("Creating Results Comparison Chart...")

fig_results = plt.figure(figsize=(14, 8))

# Model comparison data
models = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting', 'XGBoost', 'Neural\nNetwork']
auc_scores = [0.849, 0.826, 0.846, 0.828, 0.769]
roi_values = [429.9, 325.5, 447.3, 368.7, 188.2]

# Create subplots
ax1 = plt.subplot(1, 2, 1)
bars1 = ax1.bar(models, auc_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
ax1.set_ylim(0.7, 0.9)
ax1.set_ylabel('AUC Score', fontsize=14)
ax1.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

ax2 = plt.subplot(1, 2, 2)
bars2 = ax2.bar(models, roi_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
ax2.set_ylabel('ROI (%)', fontsize=14)
ax2.set_title('Business Impact (ROI %)', fontsize=16, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{height:.0f}%', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Machine Learning Model Performance & Business Impact', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/visualizations/results_comparison.png', bbox_inches='tight', facecolor='white')
plt.savefig('../outputs/visualizations/results_comparison.jpg', bbox_inches='tight', facecolor='white')
plt.close()

print("\nAll visualizations generated successfully!")
print("Check ../outputs/visualizations/ folder")