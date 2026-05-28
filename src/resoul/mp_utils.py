from __future__ import annotations

import logging
import multiprocessing
import os
import re
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment
from spacy.lang.en import English

# Use explicit spawn context to avoid CUDA issues with fork
# This ensures child processes don't inherit parent's CUDA state
_mp_context = multiprocessing.get_context("spawn")


@dataclass
class ChunkTask:
    """A single TTS work unit."""

    chapter_name: str
    chunk_index: int
    text: str
    output_path: str


@dataclass
class ChunkResult:
    """Result from processing a chunk."""

    chapter_name: str
    chunk_index: int
    output_path: str
    success: bool
    error: str | None = None


def extract_chapter_number(filename: str) -> tuple[int | float, str]:
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
    stem = Path(filename).stem.lower()

    if stem.isdigit():
        return (int(stem), filename)

    patterns = [
        r"chapter[_-]?(\d+)",
        r"chap[_-]?(\d+)",
        r"ch[_-]?(\d+)",
        r"part[_-]?(\d+)",
        r"(\d+)[_-]?chapter",
        r"(\d+)[_-]?ch",
        r"(\d{1,4})$",
        r"(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return (int(match.group(1)), filename)

    return (float("inf"), filename)


def get_spacy_lang(lang):
    if lang == "en":
        return English()
    else:
        raise NotImplementedError


def split_sentence(text, lang="en", text_split_length=250):
    """Preprocess the input text into chunks respecting sentence boundaries."""
    text_splits = []
    if text_split_length is not None and len(text) >= text_split_length:
        text_splits.append("")
        nlp = get_spacy_lang(lang)
        nlp.max_length = 6000000
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        for sentence in doc.sents:
            if len(text_splits[-1]) + len(str(sentence)) <= text_split_length:
                text_splits[-1] += " " + str(sentence)
                text_splits[-1] = text_splits[-1].lstrip()
            elif len(str(sentence)) > text_split_length:
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

        if len(text_splits) > 1 and text_splits[0] == "":
            del text_splits[0]
    else:
        text_splits = [text.lstrip()]

    return text_splits


def _init_tts_patches():
    """Apply necessary patches for PyTorch 2.6+ and torchaudio compatibility."""
    import torch

    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load


def gpu_worker_loop(
    gpu_id: int,
    task_queue,  # multiprocessing.Queue
    result_queue,  # multiprocessing.Queue
    model_name: str,
    speaker_id: str,
):
    """
    Persistent worker that loads the model once and processes tasks from the queue.

    This runs in a separate process, one per GPU.
    """
    print(f"[GPU {gpu_id}] Worker process started, PID={os.getpid()}", flush=True)

    try:
        _init_tts_patches()

        import torch
        from TTS.api import TTS

        # Select the specific GPU device
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)

        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"[GPU {gpu_id}] Using {gpu_name} ({device})", flush=True)

        # Warm up CUDA on correct device
        _ = torch.zeros(1, device=device)

        print(f"[GPU {gpu_id}] Loading TTS model on CPU first...", flush=True)
        tts = TTS(model_name=model_name, progress_bar=False, gpu=False)

        # Move model to specific GPU
        print(f"[GPU {gpu_id}] Moving model to {device}...", flush=True)
        tts.synthesizer.tts_model.to(device)
        if hasattr(tts.synthesizer, 'vocoder_model') and tts.synthesizer.vocoder_model is not None:
            tts.synthesizer.vocoder_model.to(device)

        # Update synthesizer's device reference
        tts.synthesizer.device = device

        actual_device = next(tts.synthesizer.tts_model.parameters()).device
        print(f"[GPU {gpu_id}] Model on {actual_device}, ready to process tasks", flush=True)
    except Exception as e:
        print(f"[GPU {gpu_id}] FATAL: Failed to initialize: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    chunks_processed = 0
    while True:
        task: ChunkTask | None = task_queue.get()

        if task is None:
            print(f"[GPU {gpu_id}] Shutdown signal received. Processed {chunks_processed} chunks.", flush=True)
            break

        chunks_processed += 1
        try:
            tts.tts_to_file(
                text=task.text,
                file_path=task.output_path,
                speaker_wav=speaker_id,
                language="en",
            )
            if chunks_processed % 10 == 0:
                print(f"[GPU {gpu_id}] Processed {chunks_processed} chunks", flush=True)
            result = ChunkResult(
                chapter_name=task.chapter_name,
                chunk_index=task.chunk_index,
                output_path=task.output_path,
                success=True,
            )
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing chunk: {e}", flush=True)
            result = ChunkResult(
                chapter_name=task.chapter_name,
                chunk_index=task.chunk_index,
                output_path=task.output_path,
                success=False,
                error=str(e),
            )

        result_queue.put(result)


def process_tts_chapters(
    chapters_directory: Path,
    output_dir: Path,
    model: str,
    speaker: str,
    num_gpus: int = 2,
    workers_per_gpu: int = 1,
    sequential: bool = False,
) -> list[str]:
    """
    Process multiple chapters using TTS inference with persistent GPU workers.

    Each GPU loads the model once and processes multiple chunks, avoiding
    the overhead of reloading the model for each chapter.

    Chunks are processed individually to prevent OOM issues.
    Skips chapters that already have output files in output_dir.

    Args:
        sequential: If True, process chapters one at a time in order (slower but
                   outputs chapters as they complete). If False (default), process
                   all chunks across all chapters in parallel for maximum throughput.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    speaker_name = Path(speaker).stem

    chapters = [
        chapter for chapter in chapters_directory.iterdir() if chapter.is_file()
    ]
    chapters.sort(key=lambda x: extract_chapter_number(x.name)[0])

    # Check which chapters already have output files
    chapters_to_process = []
    skipped_chapters = []
    for chapter in chapters:
        chapter_name = chapter.stem
        expected_output = output_dir / f"{speaker_name}-{chapter_name}.wav"
        if expected_output.exists():
            skipped_chapters.append((chapter, expected_output))
        else:
            chapters_to_process.append(chapter)

    logging.info("Chapter processing order:")
    for i, chapter in enumerate(chapters):
        chapter_num, _ = extract_chapter_number(chapter.name)
        status = (
            "(skipped - already exists)" if chapter not in chapters_to_process else ""
        )
        logging.info(
            f"  {i + 1}. {chapter.name} (extracted number: {chapter_num}) {status}"
        )

    if skipped_chapters:
        logging.info(
            f"Skipping {len(skipped_chapters)} chapters with existing output files"
        )

    if not chapters_to_process:
        logging.info("All chapters already processed, nothing to do")
        return [str(output_dir / f"{speaker_name}-{ch.stem}.wav") for ch in chapters]

    if sequential:
        output_files = _process_chapters_sequential(
            chapters_to_process=chapters_to_process,
            output_dir=output_dir,
            speaker_name=speaker_name,
            model=model,
            speaker=speaker,
            num_gpus=num_gpus,
            workers_per_gpu=workers_per_gpu,
        )
    else:
        output_files = _process_chapters_parallel(
            chapters_to_process=chapters_to_process,
            output_dir=output_dir,
            speaker_name=speaker_name,
            model=model,
            speaker=speaker,
            num_gpus=num_gpus,
            workers_per_gpu=workers_per_gpu,
        )

    # Include already-existing files in the return list
    for _, existing_output in skipped_chapters:
        output_files.append(str(existing_output))

    return output_files


def _process_chapters_parallel(
    chapters_to_process: list[Path],
    output_dir: Path,
    speaker_name: str,
    model: str,
    speaker: str,
    num_gpus: int,
    workers_per_gpu: int,
) -> list[str]:
    """Process all chapters in parallel (chunks from all files mixed together)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        chunks_dir = tmpdir / "chunks"
        chunks_dir.mkdir()

        all_tasks: list[ChunkTask] = []
        chapter_chunk_counts: dict[str, int] = {}

        for chapter_file in chapters_to_process:
            with open(chapter_file, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            chunks = split_sentence(text, lang="en", text_split_length=399)
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

            chapter_name = chapter_file.stem
            chapter_chunk_counts[chapter_name] = len(chunks)

            for i, chunk_text in enumerate(chunks):
                task = ChunkTask(
                    chapter_name=chapter_name,
                    chunk_index=i,
                    text=chunk_text,
                    output_path=str(chunks_dir / f"{chapter_name}_chunk_{i:05d}.wav"),
                )
                all_tasks.append(task)

        total_chunks = len(all_tasks)
        logging.info(f"Created {total_chunks} chunks from {len(chapters_to_process)} chapters")

        # Sort by length for load balancing
        all_tasks.sort(key=lambda t: len(t.text), reverse=True)

        results = _run_persistent_workers(
            tasks=all_tasks,
            num_gpus=num_gpus,
            model_name=model,
            speaker_id=speaker,
            workers_per_gpu=workers_per_gpu,
        )

        failed = [r for r in results if not r.success]
        if failed:
            for f in failed:
                logging.error(f"Failed chunk: {f.chapter_name} chunk {f.chunk_index}: {f.error}")
            raise RuntimeError(f"{len(failed)} chunks failed to process")

        # Combine chunks into chapter audio files
        output_files = []
        for chapter_file in chapters_to_process:
            chapter_name = chapter_file.stem
            chunk_count = chapter_chunk_counts[chapter_name]

            chunk_files = []
            for i in range(chunk_count):
                chunk_path = chunks_dir / f"{chapter_name}_chunk_{i:05d}.wav"
                if chunk_path.exists():
                    chunk_files.append(chunk_path)

            if not chunk_files:
                logging.warning(f"No chunks found for chapter {chapter_name}")
                continue

            combined = AudioSegment.empty()
            for chunk_file in chunk_files:
                audio = AudioSegment.from_file(str(chunk_file))
                combined += audio

            output_path = output_dir / f"{speaker_name}-{chapter_name}.wav"
            combined.export(str(output_path), format="wav")
            output_files.append(str(output_path))
            logging.info(f"Created {output_path} from {len(chunk_files)} chunks")

        return output_files


def _process_chapters_sequential(
    chapters_to_process: list[Path],
    output_dir: Path,
    speaker_name: str,
    model: str,
    speaker: str,
    num_gpus: int,
    workers_per_gpu: int,
) -> list[str]:
    """Process chapters one at a time in order, outputting each before moving to next."""
    output_files = []

    # Start workers once, reuse for all chapters
    task_queues = [_mp_context.Queue() for _ in range(num_gpus)]
    result_queue = _mp_context.Queue()

    total_workers = num_gpus * workers_per_gpu
    logging.info(f"Starting {total_workers} workers ({workers_per_gpu} per GPU)")

    workers = []
    for gpu_id in range(num_gpus):
        for _ in range(workers_per_gpu):
            p = _mp_context.Process(
                target=gpu_worker_loop,
                args=(gpu_id, task_queues[gpu_id], result_queue, model, speaker),
            )
            p.start()
            workers.append(p)

    try:
        for chapter_idx, chapter_file in enumerate(chapters_to_process):
            chapter_name = chapter_file.stem
            logging.info(f"Processing chapter {chapter_idx + 1}/{len(chapters_to_process)}: {chapter_name}")

            with open(chapter_file, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            chunks = split_sentence(text, lang="en", text_split_length=399)
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Create tasks for this chapter
                tasks = []
                for i, chunk_text in enumerate(chunks):
                    task = ChunkTask(
                        chapter_name=chapter_name,
                        chunk_index=i,
                        text=chunk_text,
                        output_path=str(tmpdir / f"chunk_{i:05d}.wav"),
                    )
                    tasks.append(task)

                # Distribute tasks round-robin
                for i, task in enumerate(tasks):
                    gpu_id = i % num_gpus
                    task_queues[gpu_id].put(task)

                # Collect results
                results = []
                for _ in range(len(tasks)):
                    result = result_queue.get()
                    results.append(result)

                failed = [r for r in results if not r.success]
                if failed:
                    for f in failed:
                        logging.error(f"Failed chunk {f.chunk_index}: {f.error}")
                    raise RuntimeError(f"{len(failed)} chunks failed for chapter {chapter_name}")

                # Combine chunks in order
                combined = AudioSegment.empty()
                for i in range(len(chunks)):
                    chunk_path = tmpdir / f"chunk_{i:05d}.wav"
                    if chunk_path.exists():
                        audio = AudioSegment.from_file(str(chunk_path))
                        combined += audio

                output_path = output_dir / f"{speaker_name}-{chapter_name}.wav"
                combined.export(str(output_path), format="wav")
                output_files.append(str(output_path))
                logging.info(f"Completed {chapter_idx + 1}/{len(chapters_to_process)}: {output_path}")

    except KeyboardInterrupt:
        logging.warning("Interrupted! Terminating workers...")
        _terminate_workers(workers)
        raise
    finally:
        # Send shutdown signals
        for gpu_id in range(num_gpus):
            for _ in range(workers_per_gpu):
                task_queues[gpu_id].put(None)
        for p in workers:
            p.join(timeout=10)

    return output_files


def _terminate_workers(workers: list) -> None:
    """Terminate all worker processes."""
    for p in workers:
        if p.is_alive():
            p.terminate()
    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join()


def _run_persistent_workers(
    tasks: list[ChunkTask],
    num_gpus: int,
    model_name: str,
    speaker_id: str,
    workers_per_gpu: int = 1,
) -> list[ChunkResult]:
    """
    Run persistent GPU workers that each load the model once and process many tasks.

    Uses explicit spawn context to ensure CUDA is properly isolated per process.
    """
    task_queues = [_mp_context.Queue() for _ in range(num_gpus)]
    result_queue = _mp_context.Queue()

    # Distribute tasks across GPUs round-robin (tasks pre-sorted by length)
    for i, task in enumerate(tasks):
        gpu_id = i % num_gpus
        task_queues[gpu_id].put(task)

    total_workers = num_gpus * workers_per_gpu
    logging.info(f"Distributing {len(tasks)} chunks across {num_gpus} GPUs ({workers_per_gpu} workers each, {total_workers} total)")
    for gpu_id in range(num_gpus):
        count = sum(1 for j, _ in enumerate(tasks) if j % num_gpus == gpu_id)
        logging.info(f"  GPU {gpu_id}: {count} chunks, {workers_per_gpu} workers")

    workers = []
    for gpu_id in range(num_gpus):
        for worker_idx in range(workers_per_gpu):
            p = _mp_context.Process(
                target=gpu_worker_loop,
                args=(gpu_id, task_queues[gpu_id], result_queue, model_name, speaker_id),
            )
            p.start()
            workers.append(p)

    results: list[ChunkResult] = []
    try:
        for i in range(len(tasks)):
            result = result_queue.get()
            results.append(result)
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                logging.info(f"Progress: {i + 1}/{len(tasks)} chunks completed")

        # Normal shutdown - send one None per worker (multiple workers share each queue)
        for gpu_id in range(num_gpus):
            for _ in range(workers_per_gpu):
                task_queues[gpu_id].put(None)

        for p in workers:
            p.join()

    except KeyboardInterrupt:
        logging.warning("Interrupted! Terminating worker processes...")
        _terminate_workers(workers)
        raise

    return results
