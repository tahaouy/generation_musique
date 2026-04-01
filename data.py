from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from music21 import converter


MidiDataset = Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]


def _is_midi_file(path: Path) -> bool:
    return path.suffix.lower() in {".mid", ".midi"}


def list_midi_files(midi_folder: str = "midi_chopin") -> List[Path]:
    root = Path(midi_folder)
    if not root.exists():
        raise FileNotFoundError(f"MIDI folder not found: {root}")
    files = [path for path in root.rglob("*") if path.is_file() and _is_midi_file(path)]
    if not files:
        raise ValueError(f"No MIDI files found in: {root}")
    return sorted(files)


def parse_midi_tokens(midi_file: Path) -> List[str]:
    score = converter.parse(str(midi_file))
    tokens: List[str] = []
    for element in score.flat.notes:
        if element.isChord:
            tokens.append(".".join(str(pitch) for pitch in element.pitches))
        else:
            tokens.append(str(element.pitch))
    return tokens


def load_tokens(midi_folder: str = "midi_chopin") -> List[str]:
    all_tokens: List[str] = []
    for midi_file in list_midi_files(midi_folder):
        try:
            all_tokens.extend(parse_midi_tokens(midi_file))
        except Exception:
            continue
    if not all_tokens:
        raise ValueError("No notes could be extracted from the MIDI corpus.")
    return all_tokens


def build_vocabulary(tokens: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab = sorted(set(tokens))
    token_to_int = {token: index for index, token in enumerate(vocab)}
    int_to_token = {index: token for token, index in token_to_int.items()}
    return token_to_int, int_to_token


def build_training_arrays(
    tokens: Sequence[str], token_to_int: Dict[str, int], sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    encoded = np.array([token_to_int[token] for token in tokens], dtype=np.int32)
    if len(encoded) <= sequence_length:
        raise ValueError(
            f"Not enough tokens for sequence_length={sequence_length}. "
            f"Need more than {sequence_length}, got {len(encoded)}."
        )
    sample_count = len(encoded) - sequence_length
    inputs = np.zeros((sample_count, sequence_length), dtype=np.int32)
    targets = np.zeros(sample_count, dtype=np.int32)
    for index in range(sample_count):
        inputs[index] = encoded[index : index + sequence_length]
        targets[index] = encoded[index + sequence_length]
    return inputs, targets


def prepare_dataset(midi_folder: str = "midi_chopin", sequence_length: int = 64) -> MidiDataset:
    tokens = load_tokens(midi_folder)
    token_to_int, int_to_token = build_vocabulary(tokens)
    x_train, y_train = build_training_arrays(tokens, token_to_int, sequence_length)
    return x_train, y_train, token_to_int, int_to_token
