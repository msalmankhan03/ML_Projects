import os
import sys

# Define Project Root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from scripts.data_loader import get_data_loaders

def verify():
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} directory not found!")
        return
        
    try:
        train_loader, test_loader, classes = get_data_loaders(DATA_DIR)
        print("✅ Data Structure Verified!")
        print(f"Detected Classes: {classes}")
        print(f"Training Samples: {len(train_loader.dataset)}")
        print(f"Testing Samples: {len(test_loader.dataset)}")
        print("-" * 30)
        print("Ready to start training!")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("\nMake sure your data is in 'data/train_dir' and 'data/test_dir'")

if __name__ == "__main__":
    verify()
