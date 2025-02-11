# Resoul
Currently a wrapper around tts engines. The team is currently working on retraining xtts based architecture that outperforms it and provide that model as well.

# Setup
```
uv venv resoul --python 3.9.19
source resoul/bin/activate
pip install -e .
```

# Docker
alternatively, if you just would like to use this as a docker image simply run:
```bash
# build the container
docker build -t csaben/resoul .
# mount any volumes in order to actually persist what you generate onto your local machine
docker run -it --rm --gpus all \
    -v $localmachineinputfolder:/app/input_folder \
    -v $localmachineoutputfolder:/app/output_folder \
    -v $localmachinereferencespeakerfolder:/app/reference_speakers \
    -e COQUI_TOS_AGREED=1 \
    csaben/resoul
    # sample run using xtts
    python -m resoul xtts files input_folder/ output_folder --reference-audio-path reference_speakers/baldree/normal.wav 

```
> xtts user may want to have model on local machine and mount it to prevent redownloading every time
```
/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2
```



#### configuring defaults
Replace defaults in `.env` or just manually specify on each run

##  Models

## XTTS (non-commercial licensing)
Works out of the box

>sample usage
```bash
# for a single file or multiple files
python -m resoul xtts files $HOME/books/test $HOME/audiobooks/test --reference-audio-path $SPEAKERS/baldree/normal.wav 

# for a textual query
python -m resoul xtts query  "what is up my man? don't be a goose and just live life to the fullest" "./whatsup.wav" --reference-audio-path $SPEAKERS/baldree/silent-king.wav
```

## OpenVoice (commerical licensing)
OpenVoice V1
Download the checkpoint from [here](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip) and extract it to the `checkpoints` folder.

>sample usage
```bash
# for a single file or multiple files
python -m resoul openvoice files $HOME/books/test $HOME/audiobooks/test --reference-audio-path $SPEAKERS/baldree/normal.wav 

# for a textual query
python -m resoul openvoice query  "what is up my man? don't be a goose and just live life to the fullest" "./whatsup.wav" --reference-audio-path $SPEAKERS/baldree/silent-king.wav
```
___

# Developer Notes

Known Bugs
- the character `C` is often omitted when it is the beginning letter in
  a word (openvoice-v1 model specifically)
- sufficiently large files will TTS fine but fail during conversion (if you use a rvc library, for e.g.)
  step (e.g. 2hr chapter fails). 

