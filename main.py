import numpy as np
from data import get_notes, num_notes, prepare_sequences
from model import create_model
from generate import generate_music, create_midi


all_notes = get_notes()
notes_to_int = num_notes()
X, y = prepare_sequences(all_notes, notes_to_int)

model = create_model(X.shape[1], len(notes_to_int))

print("Entraînement du modèle...")
model.fit(X, y, epochs=50, batch_size=64)

print("Génération de musique...")
seed_sequence = X[0]  
generated_notes = generate_music(model, seed_sequence, 200)

output_file = "generated_music.mid"
create_midi([int(note) for note in generated_notes], notes_to_int, output_file)
print(f"Musique générée enregistrée sous {output_file}.")
