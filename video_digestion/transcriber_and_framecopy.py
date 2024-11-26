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
import cv2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

class MediaProcessor:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)
        self.MAX_CHUNK_SIZE = 24 * 1024 * 1024  # 24MB for API limit
        self.OUTPUT_DIR = None
        self.chunks_completed = 0
        self.total_chunks = 0
        # Fixed intervals
        self.AUDIO_CHUNK_DURATION = 60  # 60 seconds for audio
        self.FRAME_CHUNK_DURATION = 10  # 10 seconds for frames

    def get_file_duration(self, audio_path):
        """Get file duration using ffprobe"""
        logging.info(f"Getting duration for: {audio_path}")
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', audio_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            logging.info(f"Duration: {duration:.2f}s")
            return duration
        except subprocess.CalledProcessError as e:
            logging.error(f"FFprobe error: {e.stderr}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            raise

    def extract_frames(self, video_path, output_dir):
        """Extract frames from video at 10-second intervals using cv2"""
        try:
            vidcap = cv2.VideoCapture(video_path)
            if not vidcap.isOpened():
                raise Exception("Error opening video file")

            fps = vidcap.get(cv2.CAP_PROP_FPS)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            frames_extracted = 0
            for time_pos in range(0, int(duration), self.FRAME_CHUNK_DURATION):
                # Set position in milliseconds
                vidcap.set(cv2.CAP_PROP_POS_MSEC, time_pos * 1000)
                success, image = vidcap.read()
                
                if success:
                    frame_path = os.path.join(output_dir, f"frame_{time_pos}.jpg")
                    cv2.imwrite(frame_path, image)
                    frames_extracted += 1
                    if frames_extracted % 10 == 0:
                        logging.info(f"Extracted {frames_extracted} frames")
                else:
                    logging.warning(f"Failed to extract frame at {time_pos}s")

            vidcap.release()
            logging.info(f"Frame extraction complete. Total frames: {frames_extracted}")
            return True
        except Exception as e:
            logging.error(f"Error extracting frames: {str(e)}")
            return False

    def split_audio_chunk(self, audio_path, start_time, output_path):
        """Split a single audio chunk using ffmpeg"""
        try:
            # Normalize paths for Windows
            audio_path = str(Path(audio_path))
            output_path = str(Path(output_path))

            cmd = [
                'ffmpeg', '-y',
                '-i', audio_path,
                '-ss', str(start_time),
                '-t', str(self.AUDIO_CHUNK_DURATION),
                '-acodec', 'libmp3lame',
                '-ab', '64k',
                '-ac', '1',
                '-ar', '22050',
                '-af', 'highpass=f=200,lowpass=f=3000',
                output_path
            ]

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if process.returncode != 0:
                logging.error(f"FFmpeg stderr: {process.stderr}")
                return False

            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > self.MAX_CHUNK_SIZE:
                    logging.error(f"Chunk too large: {file_size / 1024 / 1024:.2f}MB")
                    return False
                elif file_size == 0:
                    logging.error("Chunk file is empty")
                    return False
                return True
            return False

        except Exception as e:
            logging.error(f"Error in split_audio_chunk: {str(e)}")
            return False

    def transcribe_chunk(self, audio_path, timestamp):
        """Transcribe a single audio chunk with timestamp"""
        try:
            start_time = time.time()
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                    prompt="This is English speech transcription."
                )
            
            self.chunks_completed += 1
            logging.info(f"Transcribed chunk {self.chunks_completed}/{self.total_chunks} in {time.time() - start_time:.2f}s")
            return timestamp, response.text
        except Exception as e:
            logging.error(f"Error transcribing chunk: {str(e)}")
            return timestamp, ""

    def process_chunk(self, audio_path, start_time):
        """Process a single chunk of the media"""
        try:
            # Create unique temp filename using timestamp
            temp_filename = f"temp_audio_{int(time.time())}_{start_time}.mp3"
            audio_chunk_path = os.path.join(self.OUTPUT_DIR, temp_filename)

            if self.split_audio_chunk(audio_path, start_time, audio_chunk_path):
                # Transcribe audio
                timestamp, transcription = self.transcribe_chunk(audio_chunk_path, start_time)
                
                # Save transcription
                transcript_path = os.path.join(self.OUTPUT_DIR, f"transcript_{timestamp}.txt")
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(f"[{timestamp}s - {timestamp + self.AUDIO_CHUNK_DURATION}s]\n")
                    f.write(transcription)

                # Clean up temporary audio file
                if os.path.exists(audio_chunk_path):
                    os.remove(audio_chunk_path)
                return True
            return False
        except Exception as e:
            logging.error(f"Error processing chunk: {str(e)}")
            if os.path.exists(audio_chunk_path):
                os.remove(audio_chunk_path)
            return False

    def parallel_process_chunks(self, audio_path, total_duration, max_workers=4):
        """Process chunks in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for start_time in range(0, int(total_duration), self.AUDIO_CHUNK_DURATION):
                future = executor.submit(
                    self.process_chunk,
                    audio_path,
                    start_time
                )
                futures.append(future)

            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error in parallel processing: {str(e)}")

    def process_media(self, video_path, audio_path):
        """Process video and audio files into synchronized chunks with parallel processing"""
        try:
            # Create output directory
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.OUTPUT_DIR = Path(f"video_digestion/output_{video_name}")
            self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            logging.info(f"Processing media files to: {self.OUTPUT_DIR}")

            # Get total duration from audio file
            total_duration = self.get_file_duration(audio_path)
            self.total_chunks = math.ceil(total_duration / self.AUDIO_CHUNK_DURATION)

            # Extract frames at 10-second intervals
            if not self.extract_frames(video_path, self.OUTPUT_DIR):
                raise Exception("Frame extraction failed")

            # Process audio chunks in parallel at 60-second intervals
            logging.info(f"Starting parallel processing of {self.total_chunks} chunks...")
            self.parallel_process_chunks(audio_path, total_duration)

            logging.info("Media processing complete")
            return str(self.OUTPUT_DIR)

        except Exception as e:
            logging.error(f"Error during media processing: {str(e)}")
            return None


def main():
    # Replace with your OpenAI API key or set it as an environment variable
    api_key = "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"

    # Initialize the processor
    processor = MediaProcessor(api_key)

    # File paths
    video_path = r"C:\Users\Filer1\dwhelper\Systems Engineering Applications Theory Lecture 01-OCT-24 0900-01.mp4"
    audio_path = r"C:\Users\Filer1\dwhelper\Systems Engineering Applications Theory Lecture 01-OCT-24 0900-01.mp3"

    # Process the media files
    output_dir = processor.process_media(video_path, audio_path)
    
    if output_dir:
        print(f"\nProcessing complete! Output saved in: {output_dir}")
    else:
        print("\nProcessing failed. Check the logs for details.")


if __name__ == "__main__":
    main()
