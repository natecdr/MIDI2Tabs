# MIDI to Guitar tabs
Converts a midi file to ASCII guitar tabs.

It mainly uses a graph theory oriented approach as of now.

## Using the tool

First, clone the repo.

Then, place your midi files in the *midis* folder.

To convert the midis to tabs, make sure the repo is the working directory and use : `python midi_tabs.py yourmidifile.mid` in the terminal.

You should find the generated tabs in the *tabs* folder.

## Expected results

This is the kind of result you are expecting to get :

```text
E ||----------------3----|5----5--------------|1----1----0----0----|--------------------|------3----1----1----|0----0--------------|------3----1----1----|0----0--------------|----------------3----|5----5--------------|1----1----0----0----|--------------------|
B ||1----1---------------|--------------------|--------------------|------3----1--------|---------------------|--------------------|---------------------|--------------------|1----1---------------|--------------------|--------------------|------3----1--------|
G ||----------12---------|5---------12--------|2---------0---------|0-------------------|12---------2---------|0---------0---------|12---------2---------|0---------0---------|----------12---------|5---------12--------|2---------0---------|0-------------------|
D ||2---------10---------|----------10--------|--------------------|12---------2--------|10-------------------|----------12--------|10-------------------|----------12--------|2---------10---------|----------10--------|--------------------|12---------2--------|
A ||3--------------------|--------------------|----------3---------|-----------3--------|---------------------|--------------------|---------------------|--------------------|3--------------------|--------------------|----------3---------|-----------3--------|
E ||----------8----------|1---------8---------|1-------------------|3-------------------|8----------1---------|3---------3---------|8----------1---------|3---------3---------|----------8----------|1---------8---------|1-------------------|3-------------------|

```
 This is the generated tab for a twinkle twinkle little star midi.
 
 ## Limitations
 The generated tabs are often quite different from how a human would play.
 
 Also, the tool can't handle complicated multi-channel MIDI files.
