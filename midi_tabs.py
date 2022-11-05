import pretty_midi
from app.tab import Tab
from app.theory import Tuning
import argparse
import traceback

def init_parser():
    parser = argparse.ArgumentParser(description="MIDI to Guitar Tabs convertor")
    parser.add_argument("source", metavar="src", type=str, help = "Name of the MIDI file to convert")
    return parser

if __name__ == "__main__":
  parser = init_parser()
  args = parser.parse_args()
  file = args.source

  if not file.endswith(".mid"):
    file += ".mid"

  try:
    f = pretty_midi.PrettyMIDI("./midis/" + file)
    tab = Tab(file[:-4], Tuning(), f)
    tab.populate()
    tab.gen_tab()
    tab.to_ascii()
    tab.to_json()
    
  except Exception as e:
    print(traceback.print_exc())
    print("There was an error. You might want to try another MIDI file. The tool tends to struggle with more complicated multi-channel MIDI files.")