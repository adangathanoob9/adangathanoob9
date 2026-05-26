from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test,
                   save_path='outputs/charts/confusion_matrix.png'):

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Dissatisfied', 'Satisfied']))

    # Confusion Matrix Chart
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Dissatisfied', 'Satisfied'],
                yticklabels=['Dissatisfied', 'Satisfied'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Confusion matrix saved to {save_path}")

    return y_pred, acc