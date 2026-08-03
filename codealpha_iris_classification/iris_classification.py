import pandas as pd
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

    # 1. Load the Dataset from downloaded CSV file
    try:
        df = pd.read_csv("task1/Iris.csv")
        print("\n[INFO] Dataset successfully loaded from local CSV file!")
    except FileNotFoundError:
        print("\n[ERROR] 'Iris.csv' not found! Please download the dataset and place it in this folder.")
        return

    print(f"Total samples: {df.shape[0]}, Features: {df.shape[1]}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("-" * 60)

    # 2. Define Features (X) and Target (y)
    # Adjust column names based on your downloaded CSV columns (e.g., 'Species' or 'target')
    # Typically, columns are: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
    
    # Dropping ID and Species columns for features X
    X = df.drop(columns=['Id', 'Species'] if 'Id' in df.columns else ['Species'])
    y = df['Species']

    # 3. Split the Data (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scale the Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train and Evaluate Multiple Models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=3)
    }

    target_names = df['Species'].unique()

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
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("-" * 60)

if __name__ == "__main__":
    main()