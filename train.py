import os
# --- MAC OVERRIDE FIX FOR TENSORFLOW CRASH ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# ---------------------------------------------

from src.data_loader import get_preprocessed_data
from src.models import build_baseline_model_4

def run_training():
    print(" Fetching and preprocessing Reuters dataset...")
    # Load the data using the data_loader module we built
    (x_train, y_train), (x_val, y_val), _ = get_preprocessed_data()
    
    print("🏗️ Building Conv1D (Baseline-4) Model...")
    # Build your best-performing baseline architecture
    model = build_baseline_model_4()
    
    print("🏋️ Training model for 10 epochs...")
    # Train the network
    model.fit(
        x_train, 
        y_train, 
        batch_size=64, 
        epochs=10, 
        validation_data=(x_val, y_val)
    )
    
    # Create an export directory to save the trained artifact
    os.makedirs("models/export", exist_ok=True)
    model.save("models/export/reuters_model.keras")
    print("Model successfully trained and saved to 'models/export/reuters_model.keras'!")

if __name__ == "__main__":
    run_training()