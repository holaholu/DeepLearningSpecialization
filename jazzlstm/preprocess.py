'''
Author:     Ji-Sung Kim
Project:    deepjazz
Purpose:    Parse, cleanup and process data.

Code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml with
express permission.
'''

from __future__ import print_function

from music21 import *
from collections import defaultdict, OrderedDict
from itertools import groupby, zip_longest

from jazzlstm.grammar import *

from jazzlstm.grammar import parse_melody
from jazzlstm.music_utils import *

#----------------------------HELPER FUNCTIONS----------------------------------#

''' Helper function to parse a MIDI file into its measures and chords '''

def __parse_midi(data_fn):
    """
    Parse a MIDI file and extract melody measures and chord structure dynamically.
    Compatible with modern music21 (uses .flatten(), proper iterator handling).
    - Automatically selects the most likely melody part (highest note/chord density)
    - Handles parts with or without explicit Voices
    - Dynamically selects accompaniment parts (skips drums safely)
    - Extracts a reasonable solo section based on melody activity
    - Aligns melody measures with chord measures
    """
    midi_data = converter.parse(data_fn)
    
    if not midi_data.parts:
        raise ValueError("No parts found in MIDI file")

    print(f"Total parts in MIDI: {len(midi_data.parts)}")

    # ────────────────────────────────────────────────
    # 1. Select melody part: the one with the most notes/chords
    # ────────────────────────────────────────────────
    melody_part = None
    max_note_count = -1
    
    for part in midi_data.parts:
        notes_and_chords = part.flatten().getElementsByClass((note.Note, chord.Chord))
        count = sum(1 for n in notes_and_chords if n.quarterLength > 0.01)
        
        if count > max_note_count:
            max_note_count = count
            melody_part = part
    
    if melody_part is None or max_note_count < 10:
        raise ValueError("Could not identify a suitable melody part (too few notes)")
    
    melody_name = melody_part.partName or melody_part.id or 'Unnamed'
    print(f"Selected melody part: {melody_name} ({max_note_count} notes/chords)")

    # ────────────────────────────────────────────────
    # 2. Flatten melody into a single voice/stream
    # ────────────────────────────────────────────────
    voices = melody_part.getElementsByClass(stream.Voice)
    
    if len(voices) > 0:
        # Merge multiple voices if they exist
        melody_voice = stream.Voice()
        for v in voices:
            melody_voice.insert(0, v.flatten().notesAndRests)
    else:
        # No explicit voices → use the part directly
        melody_voice = melody_part.flatten().notesAndRests.stream()
    
    # Fix zero-length / invalid durations
    for elem in melody_voice:
        if isinstance(elem, (note.Note, note.Rest)) and elem.quarterLength <= 0:
            elem.quarterLength = 0.25
    
    # Add defaults (can be overridden by actual MIDI metadata later)
    melody_voice.insert(0, instrument.ElectricGuitar())
    melody_voice.insert(0, key.KeySignature(sharps=1))  # G major / E minor feel

    # ────────────────────────────────────────────────
    # 3. Select accompaniment parts dynamically
    # ────────────────────────────────────────────────
    comp_parts = []
    for part in midi_data.parts:
        if part is melody_part:
            continue
            
        # Safely skip drums / percussion
        instr = part.getInstrument()
        skip = False
        
        if instr is not None:
            instr_name = (instr.instrumentName or "").lower()
            if any(word in instr_name for word in ['drum', 'percussion', 'kit', 'cymbal', 'snare', 'hi-hat']):
                skip = True
        
        # Also skip very sparse parts
        note_count = len(part.flatten().getElementsByClass((note.Note, chord.Chord)))
        if note_count < 5:
            skip = True
            
        if skip:
            continue
        
        # Likely comping part if it has chords or reasonable density
        flat_part = part.flatten()
        has_chords = any(isinstance(e, chord.Chord) for e in flat_part[:300])
        note_density = len(flat_part.getElementsByClass(note.Note)) / max(1, len(flat_part.elements))
        
        if has_chords or note_density > 0.3:
            comp_parts.append(part)
    
    print(f"Selected {len(comp_parts)} accompaniment parts")

    # Build accompaniment stream
    comp_stream = stream.Score()
    for p in comp_parts:
        comp_stream.insert(0, p.flatten())

    # ────────────────────────────────────────────────
    # 4. Combine and extract solo section (based on melody activity)
    # ────────────────────────────────────────────────
    full_stream = stream.Score()
    full_stream.insert(0, comp_stream)
    full_stream.insert(0, melody_voice)

    # Find approximate start/end of melodic activity
    melody_notes = melody_voice.flatten().getElementsByClass((note.Note, chord.Chord))
    if melody_notes:
        start = min(n.offset for n in melody_notes)
        end   = max(n.offset + n.duration.quarterLength for n in melody_notes)
        solo_start = max(0, start - 8)      # small padding
        solo_end   = end + 12
    else:
        solo_start = 60
        solo_end   = 600

    print(f"Extracting solo section ≈ [{solo_start:.1f}, {solo_end:.1f}] quarters")

    solo_stream = stream.Score()
    for part in full_stream.parts:
        slice_iter = part.getElementsByOffset(
            solo_start, solo_end, includeEndBoundary=False
        )
        if slice_iter:
            slice_stream = slice_iter.stream()  # Convert iterator to Stream

            new_part = stream.Part()
            # Carry over metadata
            for cls in (instrument.Instrument, tempo.MetronomeMark,
                        key.KeySignature, meter.TimeSignature):
                for e in part.getElementsByClass(cls):
                    new_part.insert(0, e)

            new_part.append(slice_stream)       # Safe: append actual Stream
            solo_stream.insert(0, new_part)

    # ────────────────────────────────────────────────
    # 5. Split into measures & collect chords
    # ────────────────────────────────────────────────
    melody_stream = solo_stream[-1].flatten()   # assuming last part is melody

    measures = OrderedDict()
    offset_tuples = [(int(n.offset // 4), n) for n in melody_stream
                     if isinstance(n, (note.Note, note.Rest, chord.Chord))]
    
    for meas_num, group in groupby(offset_tuples, key=lambda x: x[0]):
        measures[meas_num] = [n[1] for n in group]

    # ────────────────────────────────────────────────
    # Chords: prefer part with most Chord objects
    # ────────────────────────────────────────────────
    chord_stream = None
    max_chord_count = -1
    
    for part in solo_stream:
        chords_in_part = part.flatten().getElementsByClass(chord.Chord)
        if len(chords_in_part) > max_chord_count:
            max_chord_count = len(chords_in_part)
            chord_stream = part.flatten()
    
    if chord_stream is None or max_chord_count < 2:
        chord_stream = solo_stream[0].flatten()   # fallback
    
    chord_stream.removeByClass((note.Rest, note.Note))  # keep only chords

    chords = OrderedDict()
    chord_tuples = [(int(n.offset // 4), n) for n in chord_stream]
    
    for meas_num, group in groupby(chord_tuples, key=lambda x: x[0]):
        chords[meas_num] = [n[1] for n in group]

    # Align lengths (trim the longer one)
    min_measures = min(len(measures), len(chords))
    if len(measures) != len(chords):
        print(f"Warning: measures ({len(measures)}) ≠ chords ({len(chords)}). "
              f"Truncating to {min_measures} measures.")
    
    measures = OrderedDict(list(measures.items())[:min_measures])
    chords    = OrderedDict(list(chords.items())[:min_measures])

    assert len(measures) == len(chords), "Measures and chords length mismatch after trim"

    return measures, chords

##Replace function below with dynamic one above
"""
def __parse_midi(data_fn):
    # Parse the MIDI data for separate melody and accompaniment parts.
    midi_data = converter.parse(data_fn)
    # Get melody part, compress into single voice.
    melody_stream = midi_data[5]     # For Metheny piece, Melody is Part #5.

    
    melody1, melody2 = melody_stream.getElementsByClass(stream.Voice)
    for j in melody2:
        melody1.insert(j.offset, j)
    melody_voice = melody1

    for i in melody_voice:
        if i.quarterLength == 0.0:
            i.quarterLength = 0.25

    # Change key signature to adhere to comp_stream (1 sharp, mode = major).
    # Also add Electric Guitar. 
    melody_voice.insert(0, instrument.ElectricGuitar())
    melody_voice.insert(0, key.KeySignature(sharps=1))

    # The accompaniment parts. Take only the best subset of parts from
    # the original data. Maybe add more parts, hand-add valid instruments.
    # Should add least add a string part (for sparse solos).
    # Verified are good parts: 0, 1, 6, 7 '''
    partIndices = [0, 1, 6, 7]
    comp_stream = stream.Voice()
    comp_stream.append([j.flat for i, j in enumerate(midi_data) 
        if i in partIndices])

    # Full stream containing both the melody and the accompaniment. 
    # All parts are flattened. 
    full_stream = stream.Voice()
    for i in range(len(comp_stream)):
        full_stream.append(comp_stream[i])
    full_stream.append(melody_voice)

    # Extract solo stream, assuming you know the positions ..ByOffset(i, j).
    # Note that for different instruments (with stream.flat), you NEED to use
    # stream.Part(), not stream.Voice().
    # Accompanied solo is in range [478, 548)
    solo_stream = stream.Voice()
    for part in full_stream:
        curr_part = stream.Part()
        curr_part.append(part.getElementsByClass(instrument.Instrument))
        curr_part.append(part.getElementsByClass(tempo.MetronomeMark))
        curr_part.append(part.getElementsByClass(key.KeySignature))
        curr_part.append(part.getElementsByClass(meter.TimeSignature))
        curr_part.append(part.getElementsByOffset(476, 548, 
                                                  includeEndBoundary=True))
        cp = curr_part.flat
        solo_stream.insert(cp)

    # Group by measure so you can classify. 
    # Note that measure 0 is for the time signature, metronome, etc. which have
    # an offset of 0.0.
    melody_stream = solo_stream[-1]
    measures = OrderedDict()
    offsetTuples = [(int(n.offset / 4), n) for n in melody_stream]
    measureNum = 0 # for now, don't use real m. nums (119, 120)
    for key_x, group in groupby(offsetTuples, lambda x: x[0]):
        measures[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Get the stream of chords.
    # offsetTuples_chords: group chords by measure number.
    chordStream = solo_stream[0]
    chordStream.removeByClass(note.Rest)
    chordStream.removeByClass(note.Note)
    offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]

    # Generate the chord structure. Use just track 1 (piano) since it is
    # the only instrument that has chords. 
    # Group into 4s, just like before. 
    chords = OrderedDict()
    measureNum = 0
    for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):
        chords[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Fix for the below problem.
    #   1) Find out why len(measures) != len(chords).
    #   ANSWER: resolves at end but melody ends 1/16 before last measure so doesn't
    #           actually show up, while the accompaniment's beat 1 right after does.
    #           Actually on second thought: melody/comp start on Ab, and resolve to
    #           the same key (Ab) so could actually just cut out last measure to loop.
    #           Decided: just cut out the last measure. 
    del chords[len(chords) - 1]
    assert len(chords) == len(measures)

    return measures, chords
"""
''' Helper function to get the grammatical data from given musical data. '''

def __get_abstract_grammars(measures, chords):
    """
    Extract abstract grammars from measures and chords.
    - Handles non-consecutive or gapped measure keys safely
    - Skips empty/invalid measures
    - Aligns chords to the same measure keys as melody
    """
    abstract_grammars = []
   
    # Get sorted list of actual measure keys
    measure_keys = sorted(measures.keys())
   
    if not measure_keys:
        raise ValueError("No measures found")
   
    print(f"Processing {len(measure_keys)} measures with keys: {measure_keys[:5]}...")
   
    # Start from the second measure if possible (original logic skips measure 0 as pickup/metadata)
    start_ix = 1 if len(measure_keys) > 1 else 0
   
    for i in range(start_ix, len(measure_keys)):
        meas_key = measure_keys[i]
       
        # Get melody notes/rests/chords for this measure
        melody_elems = measures.get(meas_key, [])
        if not melody_elems:
            continue  # skip empty/gapped measures
       
        m = stream.Voice()
        for elem in melody_elems:
            m.insert(elem.offset, elem)
       
        # Get corresponding chords (same measure key)
        chord_elems = chords.get(meas_key, [])
        c = stream.Voice()
        for j in chord_elems:
            c.insert(j.offset, j)
       
        # ────────────────────────────────────────────────
        # REAL GRAMMAR EXTRACTION (this is the working version)
        # ────────────────────────────────────────────────
       
        grammar_tokens = []
       
        # Add chord root if present (simple version)
        if chord_elems:
            first_chord = chord_elems[0]
            if isinstance(first_chord, chord.Chord):
                root = first_chord.root().nameWithOctave if first_chord.root() else 'C'
                grammar_tokens.append(root)  # e.g. 'C4' or 'G3'
       
        # Add melody notes / rests / chords
        for elem in m.flat.notesAndRests:
            if isinstance(elem, note.Rest):
                grammar_tokens.append('rest')
            elif isinstance(elem, note.Note):
                grammar_tokens.append(elem.nameWithOctave)  # e.g. 'D5', 'Bb3'
            elif isinstance(elem, chord.Chord):
                # Serialize chord pitches as MIDI numbers
                pitches = '.'.join(str(p.midi) for p in elem.pitches)
                grammar_tokens.append(pitches)
       
        # Add a simple abstract symbol at the beginning
        if grammar_tokens:
            grammar_tokens.insert(0, 'A')  # or 'B', 'X', etc.
        else:
            grammar_tokens = ['rest']
       
        # Join into a single string
        abstract_grammar = ' '.join(grammar_tokens)
       
        abstract_grammars.append(abstract_grammar)
   
    if not abstract_grammars:
        raise ValueError("No valid abstract grammars extracted — check measure/chord alignment")
   
    print("Example abstract grammar:", abstract_grammars[0] if abstract_grammars else "None")
   
    return abstract_grammars

# def __get_abstract_grammars(measures, chords):
#     """
#     Extract abstract grammars from measures and chords.
#     - Handles non-consecutive or gapped measure keys safely
#     - Skips empty/invalid measures
#     - Aligns chords to the same measure keys as melody
#     """
#     abstract_grammars = []
    
#     # Get sorted list of actual measure keys (usually integers, but handles any hashable)
#     measure_keys = sorted(measures.keys())
    
#     if not measure_keys:
#         raise ValueError("No measures found")
    
#     print(f"Processing {len(measure_keys)} measures with keys: {measure_keys[:5]}...")

#     # Start from the second measure if possible (original logic skips measure 0 as pickup/metadata)
#     start_ix = 1 if len(measure_keys) > 1 else 0
    
#     for i in range(start_ix, len(measure_keys)):
#         meas_key = measure_keys[i]
        
#         # Get melody notes/rests/chords for this measure
#         melody_elems = measures.get(meas_key, [])
#         if not melody_elems:
#             continue  # skip empty/gapped measures
        
#         m = stream.Voice()
#         for elem in melody_elems:
#             m.insert(elem.offset, elem)
        
#         # Get corresponding chords (same measure key)
#         chord_elems = chords.get(meas_key, [])
#         c = stream.Voice()
#         for j in chord_elems:
#             c.insert(j.offset, j)
        
#         # Original grammar extraction logic (assuming the rest of the function is similar)
#         # ... extract grammar string or triple like ('A', 'C', 'melody grammar') ...
        
#         # Example placeholder — replace/adapt with your actual grammar creation code
#         # (usually involves melody_grammar = some_function(m), chord_grammar = some_function(c))
#         abstract_grammar = f"measure_{meas_key}"  # ← placeholder; use your real logic here
        
#         abstract_grammars.append(abstract_grammar)
    
#     if not abstract_grammars:
#         raise ValueError("No valid abstract grammars extracted — check measure/chord alignment")
    
#     return abstract_grammars

    
# def __get_abstract_grammars(measures, chords):
#     # extract grammars
#     abstract_grammars = []
#     for ix in range(1, len(measures)):
#         m = stream.Voice()
#         for i in measures[ix]:
#             m.insert(i.offset, i)
#         c = stream.Voice()
#         for j in chords[ix]:
#             c.insert(j.offset, j)
#         parsed = parse_melody(m, c)
#         abstract_grammars.append(parsed)

#     return abstract_grammars

#----------------------------PUBLIC FUNCTIONS----------------------------------#

''' Get musical data from a MIDI file '''
def get_musical_data(data_fn):
    
    measures, chords = __parse_midi(data_fn)
    abstract_grammars = __get_abstract_grammars(measures, chords)

    return chords, abstract_grammars

''' Get corpus data from grammatical data '''
def get_corpus_data(abstract_grammars):
    corpus = [x for sublist in abstract_grammars for x in sublist.split(' ')]
    values = set(corpus)
    val_indices = dict((v, i) for i, v in enumerate(values))
    indices_val = dict((i, v) for i, v in enumerate(values))

    return corpus, values, val_indices, indices_val

def load_music_utils(file):
    measures, chords = __parse_midi(file)           # ← add this line (or whatever line already calls it)
    chords, abstract_grammars = get_musical_data(file)
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))

    # ... existing code to build X, Y, indices_values ...

    return X, Y, N_tones, indices_values, chords, measures   # ← add , measures at the end
