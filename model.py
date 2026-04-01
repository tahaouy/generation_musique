from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, Input
from tensorflow.keras.optimizers import Adam


def create_model(sequence_length: int, vocab_size: int) -> Model:
    inputs = Input(shape=(sequence_length,), dtype="int32")
    x = Embedding(input_dim=vocab_size, output_dim=128)(inputs)
    x = GRU(256, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = GRU(256)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(vocab_size, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model
