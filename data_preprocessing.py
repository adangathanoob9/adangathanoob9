import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Drop unnecessary columns
    df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True, errors='ignore')

    # Fill missing values
    df['Arrival Delay in Minutes'].fillna(
        df['Arrival Delay in Minutes'].median(), inplace=True
    )

    # Encode categorical columns
    le = LabelEncoder()
    cat_cols = ['Gender', 'Customer Type',
                'Type of Travel', 'Class', 'satisfaction']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✅ Data preprocessing complete!")
    print(f"   Train size: {X_train.shape}, Test size: {X_test.shape}")

    return X_train, X_test, y_train, y_test, df