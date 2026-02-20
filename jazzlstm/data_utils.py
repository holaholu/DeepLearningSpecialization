from jazzlstm.music_utils import * 
from jazzlstm.preprocess import * 
from tensorflow.keras.utils import to_categorical
from music21 import stream, note, chord, instrument
from collections import defaultdict
from mido import MidiFile
from pydub import AudioSegment
from pydub.generators import Sine
import math

#chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
#corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
#N_tones = len(set(corpus))
n_a = 64
x_initializer = np.zeros((1, 1, 90))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def load_music_utils(file):
    chords, abstract_grammars = get_musical_data(file)
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    X, Y, N_tones = data_processing(corpus, tones_indices, 60, 30)   
    return (X, Y, N_tones, indices_tones, chords)


def generate_music(inference_model, indices_tones, chords):
    """
    Generate a music21 stream from the inference model predictions,
    combining generated melody with the chord accompaniment.
    
    Args:
        inference_model: the trained inference Keras model
        indices_tones: dict {index: tone_string}
        chords: dict {measure_number: list of music21.chord.Chord objects}
    
    Returns:
        music21 Score object containing chords + generated melody
    """
    # Re-index chords to 0,1,2,... so we can safely use range(len(chords))
    sorted_keys = sorted(chords.keys())
    rekeyed_chords = {}
    for new_i, old_key in enumerate(sorted_keys):
        rekeyed_chords[new_i] = chords[old_key]
    chords = rekeyed_chords

    print(f"Number of measures with chords: {len(chords)}")

    # Prepare initial inputs for inference
    n_values = len(indices_tones)
    x_initializer = np.zeros((1, 1, n_values))
    a_initializer = np.zeros((1, 64))   # assuming n_a = 64
    c_initializer = np.zeros((1, 64))

    # Generate sequence
    results, indices = predict_and_sample(
        inference_model,
        x_initializer=x_initializer,
        a_initializer=a_initializer,
        c_initializer=c_initializer
    )

    # Create output score
    out_stream = stream.Score()

    # ======================================
    # 1. Add chord accompaniment
    # ======================================
    curr_offset = 0.0
    for i in range(len(chords)):
        measure_chords = stream.Voice()
        for chord_obj in chords[i]:
            # Keep original offset within measure (mod 4 is common in jazz assignment)
            measure_chords.insert(chord_obj.offset % 4, chord_obj)
        
        measure_chords.offset = curr_offset
        out_stream.insert(curr_offset, measure_chords)
        curr_offset += 4.0  # advance by one measure (4/4)

    # ======================================
    # 2. Add generated melody
    # ======================================
    melody_part = stream.Part()
    melody_part.insert(instrument.ElectricGuitar())  # or Piano, TenorSax, etc.

    for idx_array in indices:           # indices shape is usually (Ty, 1)
        idx = idx_array[0]
        if idx not in indices_tones:
            print(f"Warning: index {idx} not found in indices_tones → skipping")
            continue

        token = indices_tones[idx].strip()

        try:
            if token == 'rest':
                elem = note.Rest()
            elif '.' in token:
                # MIDI pitch numbers chord → e.g. "60.64.67"
                pitches = [int(p) for p in token.split('.')]
                elem = chord.Chord(pitches)
            elif ':' in token or token.endswith(('maj', 'min', 'dim', 'aug', '7', '9', '11')):
                # Chord symbol → e.g. "C:maj7", "Bb:min"
                elem = chord.Chord(token)
            else:
                # Single note → e.g. "C4", "F#5", "Bb3"
                elem = note.Note(token)

            elem.quarterLength = 0.5   # ← adjust this value to change note duration
            melody_part.append(elem)

        except Exception as e:
            print(f"Skipping invalid token '{token}': {e}")
            rest = note.Rest()
            rest.quarterLength = 0.5
            melody_part.append(rest)

    out_stream.insert(0, melody_part)

    return out_stream
    
def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    Ty -- length of the sequence you'd like to generate.
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    ### START CODE HERE ###
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    indices = np.argmax(pred, axis = -1)
    results = to_categorical(indices, num_classes=90)
    ### END CODE HERE ###
    
    return results, indices


def note_to_freq(note, concert_A=440.0):
  '''
  from wikipedia: http://en.wikipedia.org/wiki/MIDI_Tuning_Standard#Frequency_values
  '''
  return (2.0 ** ((note - 69) / 12.0)) * concert_A

def ticks_to_ms(ticks, tempo, mid):
    tick_ms = math.ceil((60000.0 / tempo) / mid.ticks_per_beat)
    return ticks * tick_ms

def mid2wav(file):
    mid = MidiFile(file)
    output = AudioSegment.silent(mid.length * 1000.0)

    tempo = 130 # bpm

    for track in mid.tracks:
        # position of rendering in ms
        current_pos = 0.0
        current_notes = defaultdict(dict)

        for msg in track:
            current_pos += ticks_to_ms(msg.time, tempo, mid)
            if msg.type == 'note_on':
                if msg.note in current_notes[msg.channel]:
                    current_notes[msg.channel][msg.note].append((current_pos, msg))
                else:
                    current_notes[msg.channel][msg.note] = [(current_pos, msg)]


            if msg.type == 'note_off':
                start_pos, start_msg = current_notes[msg.channel][msg.note].pop()

                duration = math.ceil(current_pos - start_pos)
                signal_generator = Sine(note_to_freq(msg.note, 500))
                #print(duration)
                rendered = signal_generator.to_audio_segment(duration=duration-50, volume=-20).fade_out(100).fade_in(30)

                output = output.overlay(rendered, start_pos)

    output.export("./output/rendered.wav", format="wav")