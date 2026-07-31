import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("=" * 60)
    print(" CODEALPHA TASK: IRIS FLOWER CLASSIFICATION")
    print("=" * 60)

    # 1. Load the Dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map(lambda x: iris.target_names[x])

    print("\n[INFO] Dataset successfully loaded!")
    print(f"Total samples: {df.shape[0]}, Features: {df.shape[1]-2}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("-" * 60)

    # 2. Define Features (X) and Target (y)
    X = iris.data
    y = iris.target

    # 3. Split the Data (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scale the Features (Best Practice for ML models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train and Evaluate Multiple Models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=3)
    }

    for name, model in models.items():
        print(f"\n--- Training Model: {name} ---")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict on test data
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc * 100:.2f}%")
        
        # Detailed Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))
        
        # Confusion Matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("-" * 60)

if __name__ == "__main__":
    main()