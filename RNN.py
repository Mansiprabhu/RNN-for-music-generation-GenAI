# Required Libraries
import numpy as np
import tensorflow as tf
from music21 import converter, instrument, note, chord, stream

# Step 1: Load and preprocess the MIDI file
def load_midi(file_path):
    midi = converter.parse(file_path)
    notes = []
    for element in midi.flat.notes:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

file_path = "/content/mus.mid"
notes = load_midi(file_path)

# Step 2: Prepare sequences for the RNN
sequence_length = 100

def prepare_sequences(notes, sequence_length):
    pitch_names = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(pitch_names)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[note] for note in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(pitch_names))
    network_output = tf.keras.utils.to_categorical(network_output)

    return network_input, network_output, note_to_int, pitch_names

network_input, network_output, note_to_int, pitch_names = prepare_sequences(notes, sequence_length)

# Step 3: Build the RNN model
def build_model(network_input, output_size):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True
        ),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(512, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(512),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = build_model(network_input, len(pitch_names))

# Step 4: Train the model
model.fit(network_input, network_output, epochs=50, batch_size=64)

# Step 5: Generate music
def generate_music(model, network_input, note_to_int, pitch_names, sequence_length, num_notes=500):
    int_to_note = {number: note for note, number in note_to_int.items()}
    start = np.random.randint(0, len(network_input) - 1)

    pattern = network_input[start]
    prediction_output = []

    for note_index in range(num_notes):
        prediction_input = np.reshape(pattern, (1, sequence_length, 1))
        prediction_input = prediction_input / float(len(pitch_names))

        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern, index)
        pattern = pattern[1:]

    return prediction_output

prediction_output = generate_music(model, network_input, note_to_int, pitch_names, sequence_length)

# Step 6: Convert output to MIDI
def create_midi(prediction_output, output_path="output.mid"):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ("." in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_path)

create_midi(prediction_output)

