import numpy as np
import ssl
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_preprocessed_data(vocab_size=10000, maxlen=200, val_frac=0.2):
    """
    Loads the Reuters dataset, pads the sequences, and creates a validation split.
    Includes an SSL context bypass to resolve macOS certificate verification errors.
    """
    # --- MAC SSL CERTIFICATE BYPASS PATCH ---
    ssl._create_default_https_context = ssl._create_unverified_context
    # ----------------------------------------
    
    # 1. Load the top 10,000 most frequent words
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=vocab_size)
    
    # 2. Pad sequences to fixed length (maxlen=200)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    
    # 3. Create validation data split (20%)
    num_val = int(len(x_train) * val_frac)
    x_val = x_train[:num_val]
    y_val = y_train[:num_val]
    
    # 4. Extract the partial training data left over
    x_train_partial = x_train[num_val:]
    y_train_partial = y_train[num_val:]
    
    return (x_train_partial, y_train_partial), (x_val, y_val), (x_test, y_test)