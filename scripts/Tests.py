import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Sample data loading function
def load_data():
    # Replace with your actual data source
    data = {
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        'churn': [0, 0, 1, 1, 0, 1, 0, 1]  # 0 = No churn, 1 = Churn
    }
    return pd.DataFrame(data)

# Main function for churn prediction
def main():
    # Load data
    df = load_data()
    
    # Features and target variable
    X = df[['age', 'salary']]
    y = df['churn']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()