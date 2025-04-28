import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import random
import time
import datetime
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for Windows
colorama.init()

# Step 1: Generate Enhanced Synthetic Banking Transaction Data
def generate_data(n=1000):
    np.random.seed(42)
    
    data = pd.DataFrame({
        'amount': np.random.exponential(scale=100, size=n),
        'transaction_freq': np.random.poisson(lam=5, size=n),  # Transaction frequency
        'location_change': np.random.randint(0, 2, size=n),
        'device_change': np.random.randint(0, 2, size=n),
        'ip_change': np.random.randint(0, 2, size=n),
        'merchant_type': np.random.randint(0, 5, size=n),
        'hour_of_day': np.random.randint(0, 24, size=n),
        'geo_velocity': np.random.uniform(0, 1000, size=n),  # Distance between transactions
        'auth_type': np.random.randint(0, 3, size=n),  # Authentication type
    })
    
    # Sophisticated fraud probability calculation
    fraud_probability = np.zeros(n)
    
    # Transaction patterns
    fraud_probability += 0.2 * (data['amount'] > 200)
    fraud_probability += 0.3 * ((data['location_change'] == 1) & (data['device_change'] == 1))
    fraud_probability += 0.15 * ((data['hour_of_day'] >= 23) | (data['hour_of_day'] <= 4))
    fraud_probability += 0.25 * (data['geo_velocity'] > 800)  # Suspicious travel speed
    fraud_probability += 0.2 * (data['transaction_freq'] > 10)  # Unusual frequency
    fraud_probability += 0.1 * (data['merchant_type'] == 4)
    
    fraud_probability = np.minimum(fraud_probability, 0.95)
    data['is_fraud'] = np.random.binomial(1, fraud_probability)
    
    return data

# Step 2: Enhanced Preprocessing
def preprocess(data):
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    
    # Feature engineering
    X['hour_sin'] = np.sin(2 * np.pi * X['hour_of_day']/24)
    X['hour_cos'] = np.cos(2 * np.pi * X['hour_of_day']/24)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Preserve 95% variance
    X_pca = pca.fit_transform(X_scaled)
    
    # Balance dataset
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_pca, y)
    
    return train_test_split(X_res, y_res, test_size=0.2, random_state=42), scaler, pca

# Step 3: Enhanced SVM Model with Grid Search
def train_model(X_train, y_train):
    param_grid = {
        'C': [1, 10, 100],
        'gamma': ['scale', 'auto', 0.1],
        'kernel': ['rbf', 'poly']
    }
    
    svm = SVC(probability=True, class_weight='balanced')
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# Step 4: Enhanced Transaction Input
def get_user_transaction():
    print(Fore.CYAN + "\n=== Enter Transaction Details ===" + Style.RESET_ALL)
    try:
        amount = float(input("Transaction amount ($): "))
        location_change = int(input("New location? (1=yes, 0=no): "))
        device_change = int(input("New device? (1=yes, 0=no): "))
        ip_change = int(input("New IP address? (1=yes, 0=no): "))
        merchant_type = int(input("Merchant type (0-4): "))
        hour_of_day = int(input("Hour of day (0-23): "))
        geo_velocity = float(input("Distance from last transaction (km): "))
        transaction_freq = int(input("Number of transactions today: "))
        auth_type = int(input("Authentication type (0=none, 1=password, 2=2FA): "))
        
        return {
            'amount': amount,
            'transaction_freq': transaction_freq,
            'location_change': location_change,
            'device_change': device_change,
            'ip_change': ip_change,
            'merchant_type': merchant_type,
            'hour_of_day': hour_of_day,
            'geo_velocity': geo_velocity,
            'auth_type': auth_type
        }
    except ValueError:
        print(Fore.RED + "Invalid input. Using random values." + Style.RESET_ALL)
        return None

# Step 5: Real-time fraud detection simulation with interactive user interface
def simulate_realtime(model, scaler, pca, n=5):
    """
    Simulates real-time fraud detection for n transactions
    Parameters:
        model: Trained SVM model
        scaler: StandardScaler for feature normalization
        pca: PCA for dimensionality reduction
        n: Number of transactions to simulate
    """
    print(Fore.GREEN + "\n🔍 Real-Time Fraud Detection System" + Style.RESET_ALL)
    
    for i in range(n):
        # Process each transaction one by one
        print(Fore.YELLOW + f"\n📊 Transaction #{i+1}" + Style.RESET_ALL)
        
        # Get transaction details from user or generate random if input is invalid
        tx = get_user_transaction()
        if tx is None:
            tx = generate_random_transaction()
        
        # Feature engineering: Convert time to cyclical features
        df_tx = pd.DataFrame([tx])
        df_tx['hour_sin'] = np.sin(2 * np.pi * df_tx['hour_of_day']/24)  # Convert hour to sine wave
        df_tx['hour_cos'] = np.cos(2 * np.pi * df_tx['hour_of_day']/24)  # Convert hour to cosine wave
        
        # Normalize and reduce dimensions using pre-trained transformers
        tx_scaled = scaler.transform(df_tx)  # Scale features to same range
        tx_pca = pca.transform(tx_scaled)    # Apply PCA transformation
        
        # Get model prediction and calculate final fraud score
        proba = model.predict_proba(tx_pca)[0]  # Get probability scores
        fraud_score = calculate_fraud_score(tx, proba[1])  # Calculate final risk score
        
        # Display analysis results with visual indicators
        display_transaction_analysis(tx, fraud_score, proba)
        
        # Trigger alert system for high-risk transactions
        if fraud_score > 0.6:  # Threshold for suspicious activity
            simulate_alert_system(tx, fraud_score)
        
        # Allow user to continue or quit
        if i < n-1 and input("\nPress Enter for next transaction or 'q' to quit: ").lower() == 'q':
            break

def generate_random_transaction():
    """
    Generates synthetic transaction data when user input is invalid
    Returns: Dictionary containing transaction features
    """
    return {
        'amount': np.random.exponential(scale=100),        # Exponential distribution for amount
        'transaction_freq': np.random.poisson(lam=5),      # Poisson distribution for frequency
        'location_change': random.randint(0, 1),           # Binary indicator for location change
        'device_change': random.randint(0, 1),             # Binary indicator for device change
        'ip_change': random.randint(0, 1),                 # Binary indicator for IP change
        'merchant_type': random.randint(0, 4),             # Categorical merchant types (0-4)
        'hour_of_day': random.randint(0, 23),              # Hour of transaction (0-23)
        'geo_velocity': random.uniform(0, 1000),           # Random distance between transactions
        'auth_type': random.randint(0, 2)                  # Authentication type (None/Password/2FA)
    }

def calculate_fraud_score(tx, base_score):
    """
    Calculates final fraud score using model prediction and business rules
    Parameters:
        tx: Transaction details
        base_score: Initial model prediction score
    Returns: Final fraud score between 0 and 1
    """
    score = base_score
    # Apply business rules to adjust the score
    if tx['amount'] > 1000:                               # High-value transaction
        score = max(score, 0.7)
    if tx['location_change'] and tx['device_change'] and tx['amount'] > 200:  # Multiple risk factors
        score = max(score, 0.8)
    if tx['geo_velocity'] > 800:                          # Suspicious travel speed
        score = max(score, 0.75)
    if tx['transaction_freq'] > 10:                       # Unusual transaction frequency
        score = max(score, 0.65)
    return score

def display_transaction_analysis(tx, fraud_score, proba):
    """
    Displays transaction analysis with colorful UI elements
    Parameters:
        tx: Transaction details
        fraud_score: Final calculated fraud score
        proba: Model prediction probabilities
    """
    # Display transaction details in formatted table
    print("\n" + "="*60)
    print(Fore.CYAN + "TRANSACTION ANALYSIS" + Style.RESET_ALL)
    print("="*60)
    
    # Print all transaction features with formatting
    print(f"Amount: ${tx['amount']:.2f}")
    print(f"Location Change: {'Yes' if tx['location_change'] else 'No'}")
    print(f"Device Change: {'Yes' if tx['device_change'] else 'No'}")
    print(f"IP Change: {'Yes' if tx['ip_change'] else 'No'}")
    print(f"Merchant Type: {tx['merchant_type']}")
    print(f"Time: {tx['hour_of_day']:02d}:00")
    print(f"Distance: {tx['geo_velocity']:.1f} km")
    print(f"Auth Type: {['None', 'Password', '2FA'][tx['auth_type']]}")
    
    # Visual risk analysis with progress bars
    print("\n" + "-"*60)
    print("Risk Analysis:")
    bar_length = 40
    legit_bar = int((1-fraud_score) * bar_length)
    fraud_bar = int(fraud_score * bar_length)
    
    # Display probability bars with Unicode blocks
    print(f"Legitimate [{('█'*legit_bar).ljust(bar_length)}] {1-fraud_score:.2f}")
    print(f"Fraudulent [{('█'*fraud_bar).ljust(bar_length)}] {fraud_score:.2f}")
    
    # Final decision with color coding
    if fraud_score > 0.6:
        print(Fore.RED + "\n⚠️ HIGH RISK TRANSACTION DETECTED!" + Style.RESET_ALL)
    else:
        print(Fore.GREEN + "\n✅ Transaction appears legitimate" + Style.RESET_ALL)

def simulate_alert_system(tx, fraud_score):
    """
    Simulates alert system for suspicious transactions
    Parameters:
        tx: Transaction details
        fraud_score: Final calculated fraud score
    """
    # Display alert message and actions taken
    print(Fore.RED + "\n🚨 ALERT: Suspicious Transaction Detected!" + Style.RESET_ALL)
    print(f"Risk Score: {fraud_score:.2f}")
    print("Actions taken:")
    print("✉️ Alert sent to fraud department")
    print("📱 SMS verification triggered")
    print("🔒 Transaction temporarily held")

# Main execution block
if __name__ == "__main__":
    # Initialize and run the fraud detection system
    print(Fore.CYAN + "Initializing Fraud Detection System..." + Style.RESET_ALL)
    
    # Data preparation phase
    print("Generating synthetic data...")
    data = generate_data()
    
    print("Preprocessing data...")
    (X_train, X_test, y_train, y_test), scaler, pca = preprocess(data)
    
    # Model training and evaluation
    print("Training SVM model...")
    model = train_model(X_train, y_train)
    
    # Display model performance metrics
    print("\nModel Evaluation:")
    y_pred = model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Start interactive simulation
    simulate_realtime(model, scaler, pca)
