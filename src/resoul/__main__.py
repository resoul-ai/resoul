from __future__ import annotations
import logging
import sys
from pathlib import Path
from os import getenv

import click
import torch
from TTS.api import TTS # xtts lib has non-commercial license, so resoul doesn't re-export for customizing
from dotenv import load_dotenv

from resoul.tts_api import OpenVoice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)],
)

# Default values
# Load environment variables from .env file
load_dotenv()
CKPT_BASE = getenv('CKPT_BASE')
CKPT_CONVERTER = getenv('CKPT_CONVERTER')
OUTPUT_DIR = getenv('OUTPUT_DIR', 'default_output_dir')
DEFAULT_REFERENCE_SPEAKER = getenv('DEFAULT_REFERENCE_SPEAKER')

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@click.group()
def cli():
    """Main CLI group"""
    pass

@cli.group()
def xtts():
    """XTTS voice synthesis commands"""
    pass

@cli.group()
def openvoice():
    """OpenVoice synthesis commands"""
    pass

class TTSBackend:
    """converts text to speech (files)"""
    def __call__(self, text:str, output_path: str, reference_audio_path: str, **kwargs)->None:
        ...

class OpenVoiceBackend(TTSBackend):
    def __init__(self, ckpt_base: Path, ckpt_converter: Path, device: str, output_dir: Path, **kwargs):
        self.openvoice = OpenVoice(
            ckpt_base=ckpt_base,
            ckpt_converter=ckpt_converter,
            device=device,
            output_dir=output_dir,
        )

    def __call__(self, text: str, output_path: str, reference_audio_path: str, speaker: str = "default", 
                 language: str = "English", speed: float = 1.0, encode_message: str = "@MyShell", **kwargs) -> None:
        self.openvoice.process_tts(
            text=text,
            output_filename=output_path,
            reference_speaker=reference_audio_path,
            speaker=speaker,
            language=language,
            speed=speed,
            encode_message=encode_message,
        )


class XTTSBackend(TTSBackend):
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", progress_bar: bool = False, device: bool = True, **kwargs):
        self.tts = TTS(model_name=model_name, progress_bar=progress_bar, gpu=device, **kwargs)

    def __call__(self, text: str, output_path: str, reference_audio_path: str, **kwargs)->None:
        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=reference_audio_path,
            **kwargs
        )

def process_input(tts: TTSBackend, input_path: Path, output_dir: Path, **kwargs) -> None:
    """Process input file(s) and generate speech"""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if it's a single file or directory
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob("**/*.txt"))

    if not files:
        click.echo(f"No .txt files found in {input_path}")
        return

    for file in files:
        try:
            # Read text from file
            with open(file, "r", encoding="utf-8") as f:
                # text = f.read().strip()
                text = f.read()

            if not text:
                click.echo(f"Skipping empty file: {file}")
                continue

            # Generate speech
            output_file = output_dir / f"{file.stem}.wav"
            print(output_file)
            tts(text=text, output_path=output_file, **kwargs)
            click.echo(f"Successfully generated: {output_file}")

        except Exception as e:
            click.echo(f"Error processing file {file}: {str(e)}", err=True)
            logging.exception(f"Detailed error for {file}")


@openvoice.command()
@click.argument("input-dir", required=True, type=click.Path(exists=True))#, help="Directory containing input text files")
@click.argument("output-dir", type=click.Path(), default=OUTPUT_DIR)#, help="Output directory for generated audio")
@click.option("--reference-audio-path", default=DEFAULT_REFERENCE_SPEAKER, help="Path to the reference speaker audio file")
@click.option("--speaker", default="default", help="Base speaker for initial TTS generation")
@click.option("--language", default="English", help="Language of the input text")
@click.option("--speed", default=1.0, type=float, help="Speed of speech generation (1.0 is normal)")
@click.option("--encode-message", default="@MyShell", help="Message to encode in the output audio")
@click.option("--ckpt-base", type=click.Path(), default=CKPT_BASE, help="Path to the base checkpoint directory")
@click.option("--ckpt-converter", type=click.Path(), default=CKPT_CONVERTER, help="Path to the converter checkpoint directory")
@click.option("--device", default=DEVICE, help="Device to use for processing (e.g., 'cuda:0', 'cpu')")
def files(input_dir: str, reference_audio_path: str, speaker: str, language: str, speed: float,
                  encode_message: str, output_dir: Path, ckpt_base: Path, ckpt_converter: Path, device: str) -> None:
    """Generate speech from all text files in the input directory using OpenVoice v1 model."""
    engine = OpenVoiceBackend(
        ckpt_base=ckpt_base,
        ckpt_converter=ckpt_converter,
        device=device,
        output_dir=output_dir
    )
    process_input(
        tts=engine,
        input_path=input_dir,
        output_dir=output_dir,
        reference_audio_path=reference_audio_path,
        speaker=speaker,
        language=language,
        speed=speed,
        encode_message=encode_message
    )

@openvoice.command()
@click.argument("text", required=True)
@click.argument("output-path", type=click.Path())
@click.option("--reference-audio-path", default=DEFAULT_REFERENCE_SPEAKER, help="Path to the reference speaker audio file")
@click.option("--speaker", default="default", help="Base speaker for initial TTS generation")
@click.option("--language", default="English", help="Language of the input text")
@click.option("--speed", default=1.0, type=float, help="Speed of speech generation (1.0 is normal)")
@click.option("--encode-message", default="@MyShell", help="Message to encode in the output audio")
@click.option("--ckpt-base", type=click.Path(), default=CKPT_BASE, help="Path to the base checkpoint directory")
@click.option("--ckpt-converter", type=click.Path(), default=CKPT_CONVERTER, help="Path to the converter checkpoint directory")
@click.option("--device", default=DEVICE, help="Device to use for processing (e.g., 'cuda:0', 'cpu')")
def query(text: str, output_path: str, reference_audio_path: str, speaker: str, language: str, speed: float,
         encode_message: str, ckpt_base: Path, ckpt_converter: Path, device: str) -> None:
    """Generate speech from text using OpenVoice v1 model."""
    engine = OpenVoiceBackend(
        ckpt_base=ckpt_base,
        ckpt_converter=ckpt_converter,
        device=device,
        output_dir=Path(output_path).parent
    )
    engine(
        text=text,
        output_path=output_path,
        reference_audio_path=reference_audio_path,
        speaker=speaker,
        language=language,
        speed=speed,
        encode_message=encode_message
    )

@xtts.command()
@click.argument("input-path", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path())
@click.option("--reference-audio-path", help="Path to speaker reference audio")
@click.option("--language", default="en", help="Language code")
@click.option("--device", default=DEVICE, help="Device for processing (cuda:0, cpu)")
def files(input_path: str, output_dir: str, reference_audio_path: str, language: str, device: str):
    """Generate speech using XTTS v2"""
    engine = XTTSBackend( device=device != "cpu")
    process_input(
        tts=engine,
        input_path=input_path,
        output_dir=output_dir,
        reference_audio_path=reference_audio_path,
        language=language
    )

@xtts.command()
@click.argument("text", required=True)
@click.argument("output-path", type=click.Path())
@click.option("--reference-audio-path", help="Path to speaker reference audio")
@click.option("--language", default="en", help="Language code")
@click.option("--device", default=DEVICE, help="Device for processing (cuda:0, cpu)")
def query(text: str, output_path: str, reference_audio_path: str, language: str, device: str):
    """Generate speech using XTTS v2"""
    engine = XTTSBackend(device=device != "cpu")
    engine(text, output_path=output_path, reference_audio_path=reference_audio_path, language=language)

if __name__ == "__main__":
    cli(prog_name="resoul")