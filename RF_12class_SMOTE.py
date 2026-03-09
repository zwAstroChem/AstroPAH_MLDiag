import numpy as np  # Arrays and math
import pandas as pd  # DataFrames and CSVs
from sklearn.ensemble import RandomForestClassifier  # ML model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Metrics
from imblearn.over_sampling import SMOTE  # Class balancing
from collections import Counter, defaultdict  # Data counting
import warnings  # Warning control
import os  # File/folder paths
import time  # Runtime tracking
from datetime import datetime  # Timestamps
import sys  # System/log output

# Suppress warnings
#warnings.filterwarnings('ignore')  # Hides non-critical errors

# --- Global Configuration ---
suffix = "6eV"  # Excitation energy tag
CONFIG = {  # Settings dictionary
    "use_smote": True,  # Enable SMOTE
    "dataset_suffix": suffix,  # Suffix tag
    "spectral_file": f"spectr_bin_all_{suffix}.txt",  # Training data
    "mixed_test_file": f"spectr_mixed_unseen_{suffix}.txt",  # Test mixtures
    "unseen_list_file": f"unseen_sample_list_{suffix}.txt",  # Excluded IDs
    "mol_list_file": "mol_list.csv",  # Master metadata
    "rf_params": {  # RF settings
        "n_estimators": 500,  # 500 trees
        "max_depth": 25,  # Tree depth
        "n_jobs": -1,  # Use all CPUs
        "random_state": 42,  # random seed
        "class_weight": "balanced_subsample"  # Weighting
    }
}

class Logger:  # Dual-output handler
    """Custom logger to output to both console and file"""
    def __init__(self, filename):  # Setup
        self.terminal = sys.stdout  # Console
        self.log = open(filename, "w", encoding="utf-8")  # File

    def write(self, message):  # Print logic
        self.terminal.write(message)  # To screen
        self.log.write(message)  # To log

    def flush(self):  # Buffer clear
        self.terminal.flush()  # Screen flush
        self.log.flush()  # Log flush

def create_output_directories():  # Folder setup
    """Create necessary output directories"""
    dirs = ['feature_importance', 'logs']  # Dir list
    for d in dirs:  # Loop
        os.makedirs(d, exist_ok=True)  # Create
    return dirs  # Output

def get_class_id(row):  # 12-class logic
    """12-class classification logic based on C size and Charge"""
    try:  # Error safety
        nc = int(row['C']) if not pd.isna(row['C']) else 0  # C atoms
        # Size categorization: <50, 50-100, >=100
        size_offset = 0 if nc < 50 else (4 if nc < 100 else 8)  # Offset
        
        cs = None  # Init charge
        for c_val in [-1, 0, 1, 2]:  # Charge range
            if row.get(f'Charge_{c_val}') == 1:  # Match
                cs = c_val  # Set
                break  # Exit
        if cs is None: return None  # Fail
        return int(size_offset + (cs + 1))  # 0-11 ID
    except:  # Catch all
        return None  # Fail

def load_all_data():  # Data ingestion
    """Load spectral data and return statistical summary"""
    mol_df = pd.read_csv(CONFIG["mol_list_file"])  # Metadata
    mol_to_class = {int(row['ID']): get_class_id(row) for _, row in mol_df.iterrows() if get_class_id(row) is not None}  # Map
    
    unseen_ids = set()  # Test set IDs
    if os.path.exists(CONFIG["unseen_list_file"]):  # Check
        unseen_df = pd.read_csv(CONFIG["unseen_list_file"], sep='\s+', header=None)  # Load
        unseen_ids = set(unseen_df[1].astype(int).tolist())  # Set

    # Load training spectra
    X_train, y_train = [], []  # Init lists
    with open(CONFIG["spectral_file"], 'r') as f:  # Open
        current_id, current_data = None, []  # State
        for line in f:  # Loop
            line = line.strip()  # Clean
            if line == 'END':  # Boundary
                if current_id in mol_to_class and current_id not in unseen_ids:  # Filter
                    # Column index 2 for normalized intensity in 6eV format
                    X_train.append([float(p[2]) for p in current_data])  # Features
                    y_train.append(mol_to_class[current_id])  # Target
                current_id, current_data = None, []  # Reset
            elif line.isdigit(): current_id = int(line)  # ID
            else: current_data.append(line.split())  # Points

    # Load mixed test set
    X_mixed, y_mixed, mixed_info = [], [], []  # Init test
    with open(CONFIG["mixed_test_file"], 'r') as f:  # Open
        current_data, meta = [], None  # State
        for line in f:  # Loop
            line = line.strip()  # Clean
            if line == 'END':  # Boundary
                if meta:  # If exists
                    # Column index 1 for intensity in mixed format
                    X_mixed.append([float(p[1]) for p in current_data])  # Mixture
                    y_mixed.append(int(meta[0]))  # True class
                    mixed_info.append(meta)  # Info
                current_data, meta = [], None  # Reset
            elif len(line.split()) > 3: # Header row
                meta = [int(x) if i < 4 else x for i, x in enumerate(line.split())]  # Parse
            else: current_data.append(line.split())  # Points

    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int32), \
           np.array(X_mixed, dtype=np.float32), np.array(y_mixed, dtype=np.int32), mixed_info  # Arrays

def main():  # Main execution
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Unique ID
    create_output_directories()  # Dirs
    
    # Initialize logger
    log_file = f"logs/full_run_{CONFIG['dataset_suffix']}_{timestamp}.log"  # Log path
    sys.stdout = Logger(log_file)  # Redirect
    
    start_time = time.time()  # Start
    
    print("="*80)  # Rule
    print(f"Machine Learning Run Log - Dataset: {CONFIG['dataset_suffix']}")  # Dataset
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  # Time
    print("="*80)  # Rule

    # 1. Data Loading and Statistics
    X_train, y_train, X_mixed, y_mixed, mixed_info = load_all_data()  # Load
    
    print("\n[1/4] Data Statistical Summary:")  # Section
    print(f"--- Training Set (Original) ---")  # Stats
    print(f"Total samples: {len(y_train)}")  # N samples
    print(f"Feature dimensions: {X_train.shape[1]}")  # Bins
    train_dist = Counter(y_train)  # Count
    for c in range(12):  # Classes
        print(f"Class {c:2d}: {train_dist.get(c, 0):5d} samples")  # Dist

    print(f"\n--- Mixed Test Set ---")  # Test stats
    print(f"Total samples: {len(y_mixed)}")  # N test
    mixed_dist = Counter(y_mixed)  # Count
    for c in range(12):  # Classes
        print(f"Class {c:2d}: {mixed_dist.get(c, 0):5d} samples")  # Dist

    # 2. SMOTE Balancing
    if CONFIG["use_smote"]:  # Check
        print("\n[2/4] Data Balancing (SMOTE):")  # SMOTE
        smote = SMOTE(random_state=42)  # Model
        X_train, y_train = smote.fit_resample(X_train, y_train)  # Resample
        print(f"Training set size after oversampling: {len(y_train)}")  # New N
        print(f"Balanced samples per class: {Counter(y_train).get(0)}")  # Balance

    # 3. Model Parameters and Training
    print("\n[3/4] Model Parameters Configuration:")  # Config
    for k, v in CONFIG["rf_params"].items():  # Params
        print(f"{k:20}: {v}")  # Print
    
    print("\nTraining model, please wait...")  # Start train
    rf = RandomForestClassifier(**CONFIG["rf_params"])  # Model
    rf.fit(X_train, y_train)  # Training

    # 4. Results Analysis
    print("\n[4/4] Final Evaluation Results:")  # Results
    y_pred = rf.predict(X_mixed)  # Predict
    
    print("\n--- Classification Report ---")  # Report
    print(classification_report(y_mixed, y_pred))  # F1/Rec

    print("--- Accuracy by Mixed Components (n_mol) ---")  # Breakdown
    results_by_nmol = defaultdict(list)  # Group
    for i in range(len(y_mixed)):  # Loop
        nmol = mixed_info[i][1]  # Get Nmol
        results_by_nmol[nmol].append(y_pred[i] == y_mixed[i])  # Match
    
    for n in sorted(results_by_nmol.keys()):  # Complexity
        acc = np.mean(results_by_nmol[n])  # Accuracy
        print(f"Mixed molecules {n:3d} | Sample count: {len(results_by_nmol[n]):4d} | Accuracy: {acc:.4f}")  # Log

    # Save Feature Importance
    fi_df = pd.DataFrame({'Importance': rf.feature_importances_})  # Important bins
    fi_path = f"feature_importance/fi_{CONFIG['dataset_suffix']}_{timestamp}.csv"  # Path
    fi_df.to_csv(fi_path, index=False)  # Save
    
    # Save Confusion Matrix as text to log
    print("\n--- Confusion Matrix (Text Version) ---")  # Matrix
    cm = confusion_matrix(y_mixed, y_pred)  # Calculate
    print(cm)  # Print

    end_time = time.time()  # End
    print("\n" + "="*80)  # Rule
    print(f"Task completed! Total time: {end_time - start_time:.2f} seconds")  # Time
    print(f"Detailed log saved to: {log_file}")  # Path
    print(f"Feature importance saved to: {fi_path}")  # Path
    print("="*80)  # Rule

if __name__ == "__main__":  # Entry
    main()  # Run