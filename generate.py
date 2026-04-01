import numpy as np
from music21 import stream, note, chord

def _sample_with_temperature(probabilities: np.ndarray, temperature: float) -> int:
    logits = np.log(np.maximum(probabilities, 1e-8)) / max(temperature, 1e-3)
    scaled = np.exp(logits) / np.sum(np.exp(logits))
    return int(np.random.choice(len(scaled), p=scaled))


def generate_music(model, seed_sequence, generated_length: int, temperature: float = 1.0):
    generated = [int(x) for x in seed_sequence]
    sequence_length = len(generated)
    for _ in range(generated_length):
        input_seq = np.array(generated[-sequence_length:], dtype=np.int32).reshape(1, sequence_length)
        probabilities = model.predict(input_seq, verbose=0)[0]
        predicted = _sample_with_temperature(probabilities, temperature)
        generated.append(predicted)
    return generated


def create_midi(predicted_notes, int_to_notes, filename):
    output_notes = []
    offset = 0.0
    for pattern in predicted_notes:
        note_str = int_to_notes.get(int(pattern))
        if note_str is None:
            continue
        if "." in note_str:
            notes_in_chord = [note.Note(n) for n in note_str.split(".")]
            chord_obj = chord.Chord(notes_in_chord)
            chord_obj.offset = offset
            output_notes.append(chord_obj)
        else:
            note_obj = note.Note(note_str)
            note_obj.offset = offset
            output_notes.append(note_obj)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=filename)
