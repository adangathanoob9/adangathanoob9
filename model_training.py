from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def train_model(X_train, y_train, save_path='outputs/models/decision_tree.pkl'):
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)

    print("✅ Model trained and saved!")
    return model