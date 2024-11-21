import numpy as np
from music21 import stream, note, chord

def generate_music(model, seed_sequence, num_notes):
    generated = list(seed_sequence)
    for _ in range(num_notes):
        input_seq = np.array(generated[-50:]).reshape(1, 50, 1)
        input_seq = input_seq / len(seed_sequence)

        predicted = np.argmax(model.predict(input_seq, verbose=0))
        generated.append(predicted)
    return generated

def create_midi(predicted_notes, notes_to_int, filename):
    int_to_notes = {number: note for note, number in notes_to_int.items()}
    output_notes = []

    for pattern in predicted_notes:
        note_str = int_to_notes.get(pattern, None)
        if note_str:
            if '.' in note_str:  
                notes = note_str.split('.')
                chord_notes = [note.Note(n) for n in notes]  
                output_notes.append(chord.Chord(chord_notes))
            else:  
                output_notes.append(note.Note(note_str))

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)
