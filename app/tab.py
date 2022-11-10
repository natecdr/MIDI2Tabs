import traceback
import numpy as np
from pretty_midi.containers import TimeSignature
from app.theory import Measure
from app.utils import *
import networkx as nx
import json
import os
from tqdm import tqdm

class Tab:
  def __init__(self, name, tuning, midi):
    quantize(midi)
    self.name = name
    self.tuning = tuning
    self.time_signatures = midi.time_signature_changes if len(midi.time_signature_changes) > 0 else [TimeSignature(4, 4, 0)]
    self.nstrings = len(tuning.strings)
    self.measures = []
    self.midi = midi
    self.graph = self._build_complete_graph()
    self.tab = None

  def populate(self):
    for i,time_signature in enumerate(self.time_signatures):
      measure_length = measure_length_ticks(self.midi, time_signature)
      time_sig_start = time_signature.time
      time_sig_end = self.time_signatures[i+1].time if i < len(self.time_signatures)-1 else self.midi.time_to_tick(self.midi.get_end_time())
      measure_ticks = np.arange(time_sig_start, time_sig_end, measure_length)

      for imeasure, measure_tick in enumerate(measure_ticks):
        notes = []
        for instrument in get_non_drum(self.midi.instruments):
          measure_start = measure_tick
          measure_end = measure_start + measure_length
          notes = np.concatenate((notes, get_notes_between(self.midi, instrument.notes, measure_start, measure_end))) 
        notes = sort_notes_by_tick(notes)
        measure = Measure(self, imeasure, time_signature)
        measure.populate(notes, imeasure, self.midi)
        self.measures.append(measure)

  def get_all_notes(self):
    notes = self.measures[0].get_all_notes()
    for i in range(1, len(self.measures)):
      notes = np.concatenate((notes, self.measures[i].get_all_notes()), axis = 1)
    return notes
    
  def _build_complete_graph(self):
    note_map = get_all_possible_notes(self.tuning)
    
    complete_graph = nx.Graph()
    for istring, string in enumerate(note_map):
      for inote, note in enumerate(string):
        complete_graph.add_node(note, pos = (istring, inote))

    complete_graph_nodes = list(complete_graph.nodes(data=True))

    for node in complete_graph_nodes:
      for node_to_link in complete_graph_nodes:
        if not node is node_to_link:
          if node_to_link[1]['pos'][1] == 0:
            dst = 0
          else:
            dst = distance_between(node[1]['pos'], node_to_link[1]['pos'])
          complete_graph.add_edge(node[0], node_to_link[0], distance = dst)

    return complete_graph

  def to_string(self):
    res = []
    for string in self.tuning.strings:
      s = string.degree.value
      s += "||" if len(string.degree.value)>1 else " ||"
      res.append(s)

    for measure in self.tab["measures"]:
      for ievent, event in enumerate(measure["events"]):
        if "notes" in event:  
          for note in event["notes"]:
            string, fret = note["string"], note["fret"]
            res[string] += str(fret)

        next_event_timing = measure["events"][ievent + 1]["measure_timing"] if ievent < len(measure["events"]) - 1 else 1.0
        dashes_to_add = max(1, math.floor((next_event_timing - event["measure_timing"]) / (1/16)))

        res = fill_measure_str(res)

        for istring in range(self.nstrings):
          res[istring] += "-" * dashes_to_add
        
      for istring in range(self.nstrings):
        res[istring] += "|"

    return res

  def gen_tab(self):
    res = {"measures":[]}

    previous_start_time = -1
    time_sig_index = -1
    event_id = 0
    sequences = [[()]]
    # sequences_likelihoods = [1]
    sequences_likelihoods = {():1}

    for imeasure, measure in enumerate(self.measures):
      print(f"{imeasure}/{len(self.measures)}")
      res_measure = {"events":[]}

      all_notes = measure.get_all_notes()

      for timing, notes in all_notes.items():
        ts_change = False
        if notes: #if notes contains one or more notes at a specific timin
          start_time = notes[0].start
          start_time_ticks = int(self.midi.time_to_tick(start_time))

          note_arrays = []
          for note in notes:
            note = midi_note_to_note(note)
            note_arrays.append(get_notes_in_graph(self.graph, note))
          try:
            paths = find_paths(self.graph, note_arrays)
            print("Number of paths :", len(paths))

            print("Sequence likelihoods :", sequences_likelihoods)
            new_sequences = []
            for sequence in sequences:
              for next_path in paths:
                new_sequence = sequence.copy()
                # print(new_sequence)
                new_sequence.append(next_path)
                new_sequences.append(new_sequence)

            split_sequences = [new_sequences[x:x+len(sequences)] for x in range(0, len(new_sequences), len(sequences))]
            sequences = new_sequences

            print("Number of sequences :", len(sequences))
            print("All sequences :", split_sequences)

            final_sequences = {}
            new_likelihoods = {}
            print("Splitted sequences :")
            for iseq, path_sequences in enumerate(split_sequences):
              # for seq in path_sequences:
              #   print(seq)

              next_path_difficulties = [compute_path_difficulty(self.graph, seq[-1], seq[-2] if len(seq) >= 2 else (), start_time, previous_start_time) for seq in path_sequences]
              next_path_probabilities = [next_path_difficulty/np.sum(next_path_difficulties) for next_path_difficulty in next_path_difficulties]
              print("Sequence probabilities :", next_path_probabilities)
              current_likelihoods = {seq[-1]:sequences_likelihoods[seq[-2]] * next_path_probabilities[i] for i, seq in enumerate(path_sequences)}
              print("Next fingering likelihoods :", current_likelihoods)

              for i, (notes, likelihood) in enumerate(current_likelihoods.items()):
                print("Max likelihood item :", notes)
                if notes not in new_likelihoods or likelihood > new_likelihoods[notes]:
                  final_sequences[notes] = path_sequences[i]
                  new_likelihoods[notes] = likelihood

            sequences_likelihoods = new_likelihoods
            sequences = list(final_sequences.values())
            
          except Exception as e:
            print(traceback.print_exc())
            print("Note arrays :", note_arrays)
            print("Notes :", notes)
            print(midi_note_to_note(notes[0]))
            print("Measure no.", imeasure)
            print("")

          if time_sig_index+1 < len(self.time_signatures) and self.time_signatures[time_sig_index+1].time <= start_time:
            time_sig_index += 1
            ts = self.time_signatures[time_sig_index]
            ts_change = True
            
          event = self.build_event(start_time, start_time_ticks, timing, ts, ts_change, event_id)
          event_id += 1

          previous_start_time = start_time

        res_measure["events"].append(event)

      res["measures"].append(res_measure)

    self.tab = res

    final_sequence = sequences[np.argmin(sequences_likelihoods.values())]
    final_sequence.pop(0)
    self.populate_tab_notes(final_sequence)
    
  def populate_tab_notes(self, sequence):
    ievent = 0
    for measure in self.tab["measures"]:
      for ievent, event in enumerate(measure["events"]):
        for path_note in sequence[ievent]:
          string, fret = self.graph.nodes[path_note]["pos"]   
          event["notes"].append({
            "degree": str(path_note.degree),
            "octave": int(path_note.octave),
            "string": string,
            "fret": fret
            })
        ievent += 1

  def build_event(self, start_time, start_time_ticks, timing, ts, ts_change, event_id):
    event = {"id" : event_id,
            "time":start_time,
            "measure_timing":None,
            "time_ticks":start_time_ticks,
            "notes":[]}

    if ts_change:
      event["time_signature_change"] = [ts.numerator, ts.denominator]

    event["measure_timing"] = timing/ts.numerator

    return event

  def to_json(self):
    if self.tab is None:
      return
      
    json_object = json.dumps(self.tab, indent=4)

    with open(os.path.join("json", self.name + ".json"), "w") as outfile:
        outfile.write(json_object)

  def to_ascii(self):
    if self.tab is None:
      return 

    notes_str = self.to_string()

    with open(f"./tabs/{self.name}.txt","w") as file:
      for string_notes in notes_str:
        file.write(string_notes + "\n")

  def __repr__(self):
    res = ""
    notes_str = self.to_string()
    for string_notes in notes_str:
      res += string_notes + "\n"

    return res
