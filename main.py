import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data import prepare_dataset
from generate import create_midi, generate_music
from model import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Chopin-style music generator.")
    parser.add_argument("--midi-folder", default="midi_chopin", help="Path to MIDI dataset folder.")
    parser.add_argument("--sequence-length", type=int, default=64, help="Input sequence length.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum number of epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--generated-length", type=int, default=300, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--output-midi", default="generated_chopin.mid", help="Output MIDI file.")
    parser.add_argument("--model-path", default="best_model.keras", help="Path to save best model.")
    return parser.parse_args()


def main():
    args = parse_args()
    x_train, y_train, token_to_int, int_to_token = prepare_dataset(
        midi_folder=args.midi_folder,
        sequence_length=args.sequence_length,
    )
    model = create_model(sequence_length=args.sequence_length, vocab_size=len(token_to_int))
    callbacks = [
        ModelCheckpoint(filepath=args.model_path, monitor="loss", save_best_only=True),
        EarlyStopping(monitor="loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    seed_index = np.random.randint(0, len(x_train))
    seed_sequence = x_train[seed_index].tolist()
    generated_sequence = generate_music(
        model=model,
        seed_sequence=seed_sequence,
        generated_length=args.generated_length,
        temperature=args.temperature,
    )
    output_path = Path(args.output_midi)
    create_midi(generated_sequence, int_to_token, str(output_path))
    print(f"Generated MIDI saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
