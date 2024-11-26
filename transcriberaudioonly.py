# transcriber.py

from openai import OpenAI
import subprocess
import os
import math
from pathlib import Path
import concurrent.futures
import json
import time
from datetime import datetime


class WhisperTranscriber:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)
        self.MAX_CHUNK_SIZE = 24 * 1024 * 1024  # 24MB to be safe (API limit is 25MB)
        self.OUTPUT_DIR = "../transcription_chunks"
        Path(self.OUTPUT_DIR).mkdir(exist_ok=True)
        self.chunks_completed = 0
        self.total_chunks = 0

    def log_debug(self, message):
        """Print debug message with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")

    def get_file_info(self, audio_path):
        """Get file info using ffprobe"""
        self.log_debug(f"Getting file info for: {audio_path}")
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stderr:
            self.log_debug(f"FFprobe stderr: {result.stderr}")

        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        bit_rate = float(data['format'].get('bit_rate', 0))
        file_size = os.path.getsize(audio_path)

        self.log_debug(f"File size: {file_size / 1024 / 1024:.2f}MB")
        self.log_debug(f"Duration: {duration:.2f}s")
        self.log_debug(f"Bit rate: {bit_rate / 1024:.2f}kbps")

        return duration, bit_rate, file_size

    def calculate_chunk_duration(self, audio_path):
        """Calculate optimal chunk duration based on file size and bitrate"""
        duration, bit_rate, file_size = self.get_file_info(audio_path)

        if bit_rate == 0:  # If bitrate not detected, estimate from file size
            bit_rate = (file_size * 8) / duration

        # Calculate how many seconds of audio fit in MAX_CHUNK_SIZE
        # Add a 10% safety margin to account for encoding overhead
        chunk_duration = (self.MAX_CHUNK_SIZE * 8 * 0.9) / bit_rate

        self.log_debug(f"Calculated chunk duration: {chunk_duration:.2f}s")
        return chunk_duration, duration

    def split_chunk(self, audio_path, start_time, chunk_duration, output_path):
        """Split a single chunk using ffmpeg"""
        self.log_debug(f"Splitting chunk: {start_time:.1f}s to {start_time + chunk_duration:.1f}s -> {output_path}")
        start_time_split = time.time()

        # Reduce bitrate to ensure chunks stay under size limit
        cmd = [
            'ffmpeg', '-y',
            '-i', audio_path,
            '-ss', str(start_time),
            '-t', str(chunk_duration),
            '-acodec', 'libmp3lame',
            '-ab', '64k',  # Reduced bitrate for smaller file size
            '-ac', '1',  # Convert to mono
            '-ar', '22050',  # Reduced sample rate (still good for speech)
            '-af', 'highpass=f=200,lowpass=f=3000',  # Focus on speech frequencies
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.stderr:
            self.log_debug(f"FFmpeg stderr for {output_path}: {result.stderr}")

        # Verify output file size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            self.log_debug(f"Chunk size: {file_size / 1024 / 1024:.2f}MB")
            if file_size > self.MAX_CHUNK_SIZE:
                raise Exception(f"Chunk too large: {file_size / 1024 / 1024:.2f}MB")
            elif file_size == 0:
                raise Exception("Chunk file is empty")

        end_time_split = time.time()
        self.log_debug(f"Chunk split completed in {end_time_split - start_time_split:.2f}s")

        return output_path

    def parallel_split_audio(self, audio_path, max_workers=4):
        """Split audio file into chunks using parallel processing"""
        chunk_duration, total_duration = self.calculate_chunk_duration(audio_path)
        self.total_chunks = math.ceil(total_duration / chunk_duration)
        chunks = []

        self.log_debug(f"Beginning parallel split into {self.total_chunks} chunks using {max_workers} workers")
        splits_completed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(self.total_chunks):
                start_time = i * chunk_duration
                chunk_path = f"{self.OUTPUT_DIR}/chunk_{i}.mp3"

                future = executor.submit(
                    self.split_chunk,
                    audio_path,
                    start_time,
                    min(chunk_duration, total_duration - start_time),
                    chunk_path
                )
                futures.append((future, chunk_path))

            for future, chunk_path in futures:
                try:
                    future.result()
                    chunks.append(chunk_path)
                    splits_completed += 1
                    self.log_debug(f"Split progress: {splits_completed}/{self.total_chunks} chunks completed")
                except Exception as e:
                    self.log_debug(f"Error splitting chunk {chunk_path}: {str(e)}")
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)

        return sorted(chunks)

    def transcribe_chunk(self, chunk_path):
        """Transcribe a single audio chunk."""
        try:
            self.log_debug(f"Starting transcription of {chunk_path}")
            start_time = time.time()

            with open(chunk_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",  # Force English language
                    prompt="This is English speech transcription."  # Help guide the model
                )

            end_time = time.time()
            self.chunks_completed += 1

            # Extract text from response (simplified handling)
            text = response.text

            # Log progress
            self.log_debug(f"Transcription completed in {end_time - start_time:.2f}s")
            self.log_debug(f"Progress: {self.chunks_completed}/{self.total_chunks} chunks transcribed")
            self.log_debug(f"Sample output: {text[:100]}...")

            return text
        except Exception as e:
            self.log_debug(f"Error transcribing chunk {chunk_path}: {str(e)}")
            return ""

    def parallel_transcribe(self, chunk_paths, max_workers=2):
        """Transcribe chunks in parallel with rate limiting"""
        self.log_debug(f"Starting parallel transcription of {len(chunk_paths)} chunks...")
        transcriptions = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(executor.map(self.transcribe_chunk, chunk_paths))
            transcriptions.extend(futures)

        return transcriptions

    def transcribe_file(self, audio_path, output_path="transcription.txt", max_workers=4):
        """Transcribe entire audio file using parallel processing"""
        try:
            self.log_debug(f"Starting transcription process for {audio_path}")

            # Split the audio file into chunks using parallel processing
            chunk_paths = self.parallel_split_audio(audio_path, max_workers=max_workers)

            # Transcribe chunks in parallel (with limited concurrency to respect API limits)
            transcriptions = self.parallel_transcribe(chunk_paths, max_workers=2)

            # Combine all transcriptions
            final_text = " ".join(transcriptions)

            # Save the complete transcription
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)

            self.log_debug(f"Transcription completed and saved to {output_path}")

            # Cleanup chunks
            for chunk_path in chunk_paths:
                os.remove(chunk_path)

            return final_text

        except Exception as e:
            self.log_debug(f"Error during transcription: {str(e)}")
            return None


def main():
    # Replace with your OpenAI API key or set it as an environment variable
    api_key = "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"

    # Initialize the transcriber
    transcriber = WhisperTranscriber(api_key)

    # Path to your audio file
    audio_path = r"C:\Users\Filer1\dwhelper\Systems Engineering Applications Theory Group Tutorials 08-NOV-02.mp3"

    # Start transcription
    transcription = transcriber.transcribe_file(
        audio_path,
        max_workers=4  # Adjust based on your CPU cores
    )

    if transcription:
        print("Transcription successful!")
        print("\nFirst 500 characters of transcription:")
        print(transcription[:500] + "...")


if __name__ == "__main__":
    main()