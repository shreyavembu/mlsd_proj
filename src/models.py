import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Global configurations matching your Reuters data setup
VOCAB_SIZE = 10000
MAXLEN = 200
NUM_CLASSES = 46

def build_baseline_model_1():
    """
    Baseline Model 1: Embedding + GlobalAveragePooling + Dense.
    """
    inputs = keras.Input(shape=(MAXLEN,), dtype="int32")
    x = layers.Embedding(VOCAB_SIZE, 64)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="baseline_model_1")
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_baseline_model_4():
    """
    Baseline Model 4: Conv1D + GlobalMaxPooling.
    Yielded the highest baseline Macro F1-score of 0.5007.
    """
    inputs = keras.Input(shape=(MAXLEN,), dtype="int32")
    x = layers.Embedding(VOCAB_SIZE, 128)(inputs)
    x = layers.Conv1D(128, 5, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs, name="baseline_model_4")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

class TokenAndPositionEmbedding(layers.Layer):
    """
    Custom token + position embedding layer for the Transformer architecture.
    """
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = layers.Embedding(
            input_dim=maxlen, output_dim=embed_dim
        )

    def call(self, x):
        seq_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions