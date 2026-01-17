from __future__ import annotations

import logging
import multiprocessing
import os
import re
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from spacy.lang.en import English
from TTS.api import (
    TTS,  # xtts lib has non-commercial license, so resoul doesn't re-export for customizing
)


def extract_chapter_number(filename: str) -> tuple[int, str]:
    """
    Extract chapter number from filename for proper sorting.

    Handles various filename formats:
    - Pure numbers: "1", "2", "10", "100"
    - Chapter prefixes: "chapter1", "ch1", "chap01"
    - Chapter with separators: "chapter-1", "chapter_01", "ch-10"
    - Numbers in filename: "book1_chapter05", "part2_ch10"
    - Zero-padded: "001", "010", "100"

    Returns:
        tuple: (chapter_number, original_filename) for sorting
    """
    # Remove file extension
    stem = Path(filename).stem.lower()

    # Pattern 1: Pure number (most common for mp_file command)
    if stem.isdigit():
        return (int(stem), filename)

    # Pattern 2: Extract number from various chapter formats
    patterns = [
        r"chapter[_-]?(\d+)",  # chapter1, chapter-1, chapter_01
        r"chap[_-]?(\d+)",  # chap1, chap-01
        r"ch[_-]?(\d+)",  # ch1, ch-10
        r"part[_-]?(\d+)",  # part1, part-2
        r"(\d+)[_-]?chapter",  # 1chapter, 01-chapter
        r"(\d+)[_-]?ch",  # 1ch, 01-ch
        r"(\d{1,4})$",  # ends with 1-4 digits
        r"(\d+)",  # any number in filename
    ]

    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return (int(match.group(1)), filename)

    # Fallback: no number found, sort to end
    return (float("inf"), filename)


def get_spacy_lang(lang):
    if lang == "en":
        return English()
    else:
        raise NotImplementedError


def split_sentence(text, lang="en", text_split_length=250):
    """Preprocess the input text"""
    text_splits = []
    if text_split_length is not None and len(text) >= text_split_length:
        text_splits.append("")
        nlp = get_spacy_lang(lang)
        nlp.max_length = 6000000  # Set the new limit (e.g., 2 million characters)
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        for sentence in doc.sents:
            if len(text_splits[-1]) + len(str(sentence)) <= text_split_length:
                # if the last sentence + the current sentence is less than the text_split_length
                # then add the current sentence to the last sentence
                text_splits[-1] += " " + str(sentence)
                text_splits[-1] = text_splits[-1].lstrip()
            elif len(str(sentence)) > text_split_length:
                # if the current sentence is greater than the text_split_length
                for line in textwrap.wrap(
                    str(sentence),
                    width=text_split_length,
                    drop_whitespace=True,
                    break_on_hyphens=False,
                    tabsize=1,
                ):
                    text_splits.append(str(line))
            else:
                text_splits.append(str(sentence))

        if len(text_splits) > 1:
            if text_splits[0] == "":
                del text_splits[0]
    else:
        text_splits = [text.lstrip()]

    return text_splits


def process_tts_chapters(
    chapters_directory: Path,
    output_dir: Path,
    num_workers: int,
    model: str,
    speaker: str,
    num_gpus: int = 2,
    workers_per_gpu: int = 1,
    # replacements_yml_filepath: Path,
) -> list[str]:
    """
    Process multiple chapters using TTS inference.

    :param chapters_directory: The directory containing chapter files to process
    :type chapters_directory: Path
    :param output_dir: The directory where output audio files will be saved
    :type output_dir: Path
    :param num_workers: The number of worker processes to use for parallel processing
        (kept for backward compatibility, ignored when num_gpus > 1)
    :type num_workers: int
    :param model: The name of the TTS model to use
    :type model: str
    :param speaker: The ID or path of the speaker voice to use
    :type speaker: str
    :param num_gpus: The number of GPUs to use for parallel processing (default: 2)
    :type num_gpus: int
    :param workers_per_gpu: The number of worker processes per GPU (default: 1)
    :type workers_per_gpu: int
    :param replacements_yml_filepath: The path to a YAML file containing text replacement rules
    :type replacements_yml_filepath: Path
    :return: A list of paths to the generated audio files
    :rtype: list[str]
    """
    # preprocess_config = PreprocessConfig().from_yaml(replacements_yml_filepath)
    chapters = [
        chapter for chapter in chapters_directory.iterdir() if chapter.is_file()
    ]
    # Sort chapters numerically by filename using robust chapter number extraction
    chapters.sort(key=lambda x: extract_chapter_number(x.name)[0])

    # Debug logging for chapter ordering
    logging.info("Chapter processing order:")
    for i, chapter in enumerate(chapters):
        chapter_num, _ = extract_chapter_number(chapter.name)
        logging.info(f"  {i + 1}. {chapter.name} (extracted number: {chapter_num})")

    texts = {}
    for chapter_file in chapters:
        with open(chapter_file, "r", encoding="utf-8", errors="replace") as f:
            text = "".join(f.readlines())
            text = split_sentence(text, lang="en", text_split_length=399)
            text = [sentence.lstrip() for sentence in text]
            # text = preprocess(sentences=text, config=preprocess_config)
            texts[chapter_file.stem] = "\n".join(text)

    return parallel_tts_inference(
        texts, output_dir, num_workers, model, speaker, num_gpus, workers_per_gpu
    )


def parallel_tts_inference(
    texts: dict[str, str],
    output_dir: Path,
    num_workers: int,
    model_name: str,
    speaker_id: str,
    num_gpus: int = 2,
    workers_per_gpu: int = 1,
) -> list[str]:
    """
    Process multiple texts in parallel using TTS inference across multiple GPUs.

    :param texts: A dictionary of texts to process, where keys are identifiers and values are the texts
    :type texts: dict[str, str]
    :param output_dir: The directory where output audio files will be saved
    :type output_dir: Path
    :param num_workers: The number of worker processes to use for parallel processing
        (kept for backward compatibility, ignored when num_gpus > 1)
    :type num_workers: int
    :param model_name: The name of the TTS model to use
    :type model_name: str
    :param speaker_id: The ID or path of the speaker voice to use
    :type speaker_id: str
    :param num_gpus: The number of GPUs to use for parallel processing (default: 2)
    :type num_gpus: int
    :param workers_per_gpu: The number of worker processes per GPU (default: 1)
    :type workers_per_gpu: int
    :return: A list of paths to the generated audio files
    :rtype: list[str]
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    name = Path(speaker_id).stem

    multiprocessing.set_start_method("spawn", force=True)

    # Split texts across GPUs in round-robin fashion
    text_items = list(texts.items())
    gpu_batches: list[list[tuple[str, str]]] = [[] for _ in range(num_gpus)]
    for i, item in enumerate(text_items):
        gpu_batches[i % num_gpus].append(item)

    logging.info(f"Distributing {len(text_items)} items across {num_gpus} GPUs")
    for gpu_id, batch in enumerate(gpu_batches):
        logging.info(f"  GPU {gpu_id}: {len(batch)} items")

    all_futures = []
    executors: list[ProcessPoolExecutor] = []

    try:
        # Create a separate process pool for each GPU
        for gpu_id in range(num_gpus):
            executor = ProcessPoolExecutor(max_workers=workers_per_gpu)
            executors.append(executor)

            for source_file, text in gpu_batches[gpu_id]:
                future = executor.submit(
                    tts_worker,
                    text,
                    str(output_dir / f"{name}-{source_file}.wav"),
                    model_name,
                    speaker_id,
                    gpu_id,
                )
                all_futures.append(future)

        results = [future.result() for future in as_completed(all_futures)]

    finally:
        # Ensure all executors are properly shut down
        for executor in executors:
            executor.shutdown(wait=True)

    return results


def tts_worker(
    text: str, output_file: str, model_name: str, speaker_id: str, gpu_id: int = 0
) -> str:
    """
    Generate audio from text using a TTS model and save it to a file.

    :param text: The input text to be converted to speech
    :type text: str
    :param output_file: The path where the output audio file will be saved
    :type output_file: str
    :param model_name: The name of the TTS model to use
    :type model_name: str
    :param speaker_id: The ID or path of the speaker voice to use
    :type speaker_id: str
    :param gpu_id: The GPU device ID to use for this worker (default: 0)
    :type gpu_id: int
    :return: The path of the saved audio file
    :rtype: str
    """
    # Set which GPU this worker should use before loading the model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tts = TTS(model_name=model_name, progress_bar=False, gpu=True)
    logging.info("saving to %s on GPU %d", output_file, gpu_id)
    tts.tts_to_file(
        text=text,
        file_path=output_file,
        speaker_wav=speaker_id,
        language="en",
    )
    logging.info("finished saving to %s", output_file)
    return output_file
