# TensorflowMusic

The aim of this project is to create computer-generated music based on a user database of midi or 
[.mxl-files](https://www.musicxml.com/). If you are using .mxl-files, the download of the current 
[MuseScore](https://musescore.org/en) version is required. Other needed imports are 
[tensorflow](https://www.tensorflow.org/install) and 
[music21](https://web.mit.edu/music21/doc/usersGuide/usersGuide_01_installing.html#usersguide-01-installing "Installing music21").
Data is saved using [Google protocol buffers](https://developers.google.com/protocol-buffers/).

The models are mostly implemented in keras, which is included in tensorflow.
