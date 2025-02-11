import os
import re
import tempfile
import textwrap

import librosa
import numpy as np
import soundfile
import torch
from spacy.lang.en import English

from resoul.openvoice.api import BaseSpeakerTTS, ToneColorConverter
from resoul.openvoice.mel_processing import spectrogram_torch


def get_spacy_lang(lang):
    if lang == "en":
        return English()
    else:
        raise NotImplementedError


def combined_tts_preprocessing(text, language="en", text_split_length=250):
    # Step 1: Initial splitting using Method 2
    nlp = get_spacy_lang(language)
    nlp.add_pipe("sentencizer")
    doc = nlp(text)

    initial_splits = []
    current_split = ""

    for sentence in doc.sents:
        if len(current_split) + len(str(sentence)) <= text_split_length:
            current_split += " " + str(sentence)
            current_split = current_split.lstrip()
        elif len(str(sentence)) > text_split_length:
            if current_split:
                initial_splits.append(current_split)
                current_split = ""
            for line in textwrap.wrap(
                str(sentence),
                width=text_split_length,
                drop_whitespace=True,
                break_on_hyphens=False,
                tabsize=1,
            ):
                initial_splits.append(line)
        else:
            if current_split:
                initial_splits.append(current_split)
            current_split = str(sentence)

    if current_split:
        initial_splits.append(current_split)

    # Step 2: Further processing using elements from Method 1
    final_splits = []
    for split in initial_splits:
        # Separate concatenated words
        split = re.sub(r"([a-z])([A-Z])", r"\1 \2", split)

        # Add language markers (assuming we have a dictionary of language marks)
        language_marks = {"en": "EN", "fr": "FR", "de": "DE"}  # Add more as needed
        mark = language_marks.get(language.lower(), "EN")
        split = f"[{mark}]{split}[{mark}]"

        final_splits.append(split)

    return final_splits


class ModifiedToneColorConverter(ToneColorConverter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # hacky chunking to prevent preprocessing (lazy)
    def convert_by_chunk(
        self,
        audio_src_path,
        src_se,
        tgt_se,
        output_path=None,
        tau=0.3,
        message="default",
        chunk_size=10 * 16000,
    ):  # 10 seconds chunks

        hps = self.hps
        audio, sample_rate = librosa.load(audio_src_path, sr=hps.data.sampling_rate)

        with tempfile.TemporaryDirectory() as temp_dir:
            chunk_files = []
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                with torch.no_grad():
                    y = torch.FloatTensor(chunk).to(self.device).unsqueeze(0)
                    spec = spectrogram_torch(
                        y,
                        hps.data.filter_length,
                        hps.data.sampling_rate,
                        hps.data.hop_length,
                        hps.data.win_length,
                        center=False,
                    ).to(self.device)
                    spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
                    converted_chunk = (
                        self.model.voice_conversion(
                            spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau
                        )[0][0, 0]
                        .cpu()
                        .numpy()
                    )

                # Write chunk to temporary file
                chunk_file = os.path.join(temp_dir, f"chunk_{i}.npy")
                np.save(chunk_file, converted_chunk)
                chunk_files.append(chunk_file)

                # Clear GPU memory
                torch.cuda.empty_cache()

            # Concatenate chunks
            converted_audio = np.concatenate([np.load(f) for f in chunk_files])

            # Add watermark
            converted_audio = self.add_watermark(converted_audio, message)

            if output_path is None:
                return converted_audio
            else:
                soundfile.write(output_path, converted_audio, hps.data.sampling_rate)


class ModifiedBaseSpeakerTTS(BaseSpeakerTTS):
    # testing out different text processing in tts function
    def tts(self, text, output_path, speaker, language="English", speed=1.0):
        mark = self.language_marks.get(language.lower(), None)
        assert mark is not None, f"language {language} is not supported"

        texts = self.split_sentences_into_pieces(text, mark)
        # texts = combined_tts_preprocessing(text)
        # texts = spacy_processing(text=text)

        audio_list = []
        for t in texts:
            t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
            t = f"[{mark}]{t}[{mark}]"
            stn_tst = self.get_text(t, self.hps, False)
            device = self.device
            speaker_id = self.hps.speakers[speaker]
            with torch.no_grad():
                x_tst = stn_tst.unsqueeze(0).to(device)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
                sid = torch.LongTensor([speaker_id]).to(device)
                audio = (
                    self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        sid=sid,
                        noise_scale=0.667,
                        noise_scale_w=0.6,
                        length_scale=1.0 / speed,
                    )[0][0, 0]
                    .data.cpu()
                    .float()
                    .numpy()
                )
            audio_list.append(audio)
        audio = self.audio_numpy_concat(
            audio_list, sr=self.hps.data.sampling_rate, speed=speed
        )

        if output_path is None:
            return audio
        else:
            soundfile.write(output_path, audio, self.hps.data.sampling_rate)
