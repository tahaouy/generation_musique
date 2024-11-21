import os
from music21 import converter
import numpy as np 

def get_notes():
    midi_folder = "./BETH"
    all_notes = []

    for file in os.listdir(midi_folder):
        if file.endswith(".mid"):
            print(f"Chargement de {file}...")
            try:
                midi = converter.parse(os.path.join(midi_folder, file))
                for element in midi.flat.notes:
                    if element.isChord:
                        all_notes.append('.'.join(str(p) for p in element.pitches))
                    else:
                        all_notes.append(str(element.pitch))
            except Exception as e:
                print(f"Erreur lors du traitement de {file} : {e}")

    return all_notes

def num_notes():
    all_notes = get_notes()
    notes = sorted(set(all_notes))
    return {note: number for number, note in enumerate(notes)}

def prepare_sequences(all_notes, notes_to_int, sequence_length=50):
    sequences = []
    targets = []

    for i in range(len(all_notes) - sequence_length):
        seq = all_notes[i:i + sequence_length]
        target = all_notes[i + sequence_length]
        sequences.append([notes_to_int[note] for note in seq])
        targets.append(notes_to_int[target])

    X = np.array(sequences) / len(notes_to_int)
    y = np.array(targets)
    y = np.eye(len(notes_to_int))[y] 
    return X, y
