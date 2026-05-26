import matplotlib.pyplot as plt
import seaborn as sns
import os

CHARTS_DIR = 'outputs/charts/'
os.makedirs(CHARTS_DIR, exist_ok=True)

def plot_satisfaction_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='satisfaction', data=df,
                  palette=['#FF6B6B', '#4ECDC4'])
    plt.title('Satisfaction Distribution')
    plt.xticks([0, 1], ['Dissatisfied', 'Satisfied'])
    plt.tight_layout()
    plt.savefig(CHARTS_DIR + 'satisfaction_distribution.png')
    plt.show()
    print("✅ Satisfaction distribution chart saved!")

def plot_feature_importance(model, X):
    import pandas as pd
    feat = pd.Series(model.feature_importances_,
                     index=X.columns).sort_values()
    plt.figure(figsize=(10, 8))
    feat.plot(kind='barh', color='steelblue')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR + 'feature_importance.png')
    plt.show()
    print("✅ Feature importance chart saved!")

def plot_satisfaction_by_travel_type(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Type of Travel', hue='satisfaction',
                  data=df, palette=['#FF6B6B', '#4ECDC4'])
    plt.title('Satisfaction by Travel Type')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR + 'satisfaction_by_travel.png')
    plt.show()
    print("✅ Travel type chart saved!")

def plot_delay_vs_satisfaction(df):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='satisfaction', y='Departure Delay in Minutes',
                data=df, palette=['#FF6B6B', '#4ECDC4'])
    plt.title('Departure Delay vs Satisfaction')
    plt.xticks([0, 1], ['Dissatisfied', 'Satisfied'])
    plt.tight_layout()
    plt.savefig(CHARTS_DIR + 'delay_vs_satisfaction.png')
    plt.show()
    print("✅ Delay vs satisfaction chart saved!")

def plot_service_heatmap(df):
    service_cols = [
        'Inflight wifi service', 'Seat comfort',
        'Food and drink', 'Inflight entertainment',
        'On-board service', 'Cleanliness'
    ]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[service_cols].corr(), annot=True,
                cmap='coolwarm', fmt='.2f')
    plt.title('Service Ratings Correlation')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR + 'service_heatmap.png')
    plt.show()
    print("✅ Service heatmap saved!")