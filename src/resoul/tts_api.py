import os

import torch

from resoul.experiments.modified_api import (
    ModifiedBaseSpeakerTTS,
    ModifiedToneColorConverter,
)
from resoul.openvoice import se_extractor


class OpenVoice:
    def __init__(self, ckpt_base, ckpt_converter, device, output_dir):
        self.base_speaker_tts = ModifiedBaseSpeakerTTS(
            f"{ckpt_base}/config.json", device=device
        )
        self.base_speaker_tts.load_ckpt(f"{ckpt_base}/checkpoint.pth")

        self.tone_color_converter = ModifiedToneColorConverter(
            f"{ckpt_converter}/config.json", device=device
        )
        self.tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.source_se = torch.load(f"{ckpt_base}/en_default_se.pth").to(device)

    def process_tts(
        self,
        text: str,
        output_filename: str,
        reference_speaker: str,
        speaker: str,
        language: str,
        speed: float,
        encode_message: str,
    ) -> str:
        target_se, _ = se_extractor.get_se(
            reference_speaker,
            self.tone_color_converter,
            target_dir="processed",
            vad=True,
        )

        save_path = os.path.join(self.output_dir, output_filename)
        src_path = os.path.join(self.output_dir, "tmp.wav")

        self.base_speaker_tts.tts(
            text, src_path, speaker=speaker, language=language, speed=speed
        )

        self.tone_color_converter.convert_by_chunk(
            audio_src_path=src_path,
            src_se=self.source_se,
            tgt_se=target_se,
            output_path=save_path,
            message=encode_message,
        )

        return save_path
