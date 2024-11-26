import os
import base64
import json
import logging
import re  # Added missing re import
from pathlib import Path
import anthropic
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

class TranscriptManager:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.transcripts = {}
        self._load_transcripts()

    def _load_transcripts(self):
        """Load all transcripts from the output directory"""
        for file in self.output_dir.glob("transcript_*.txt"):
            try:
                timestamp = int(file.stem.split('_')[1])
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.transcripts[timestamp] = content
            except Exception as e:
                logging.error(f"Error loading transcript {file}: {str(e)}")

    def get_context_window(self, timestamp, window_size=300):
        """Get transcript context for ±5 minutes around timestamp"""
        context = []
        start_time = max(0, timestamp - window_size)
        end_time = timestamp + window_size

        # Sort timestamps to maintain chronological order
        for ts in sorted(self.transcripts.keys()):
            if start_time <= ts <= end_time:
                context.append(self.transcripts[ts])

        return "\n".join(context)

class FrameAnalyzer:
    def __init__(self, api_key, output_dir, context_description):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.context_description = context_description
        self.transcript_manager = TranscriptManager(output_dir)
        self.analysis_dir = self.output_dir / "frame_analysis"
        self.analysis_dir.mkdir(exist_ok=True)

    def _encode_image(self, image_path):
        """Encode image to base64"""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _get_frames_for_minute(self, minute):
        """Get all frames within a minute period"""
        frames = []
        start_time = minute * 60
        end_time = start_time + 60
        
        for second in range(start_time, end_time, 10):  # 10-second intervals
            frame_path = self.output_dir / f"frame_{second}.jpg"
            if frame_path.exists():
                frames.append((second, frame_path))
        
        return frames

    def _parse_claude_response(self, response):
        """Parse Claude's response and extract meaningful information"""
        try:
            # Assuming the response content is a list of text blocks
            full_text = ' '.join([block.text for block in response.content if hasattr(block, 'text')])
            
            # Extract frames section
            frames_match = re.search(r'<frames>(.*?)</frames>', full_text, re.DOTALL)
            frames_content = frames_match.group(1) if frames_match else "No frame details found"
            
            # Extract connections section
            connections_match = re.search(r'<connections>(.*?)</connections>', full_text, re.DOTALL)
            connections_content = connections_match.group(1) if connections_match else "No audio-visual connections found"
            
            return {
                "frames": frames_content.strip(),
                "audio_visual_connections": connections_content.strip()
            }
        except Exception as e:
            logging.error(f"Error parsing Claude response: {str(e)}")
            return {
                "frames": "Parsing error",
                "audio_visual_connections": "Parsing error"
            }

    def analyze_minute_segment(self, minute):
        """Analyze all frames within a minute segment"""
        frames = self._get_frames_for_minute(minute)
        if not frames:
            logging.warning(f"No frames found for minute {minute}")
            return

        # Get transcript context
        transcript_context = self.transcript_manager.get_context_window(minute * 60)
        
        # Prepare messages for each frame
        results = []
        for second, frame_path in frames:
            try:
                # Create message for Claude
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Frame {second}:"
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": self._encode_image(frame_path)
                                }
                            },
                            {
                                "type": "text",
                                "text": f"\n\nTranscription Context:\n{transcript_context}\n\nContext: {self.context_description}"
                            }
                        ]
                    }
                ]

                # Make API call
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    temperature=0,
                    messages=messages,
                    system="The assistant processes video frames and audio transcripts sequentially, analyzing their relationships and meaningful connections.\n\nFRAME ANALYSIS:\nSkip frame and output \"null\" if containing:\n- Desktop/system interfaces\n- Application windows\n- Menus/toolbars\n- Non-informational overlays\n\nOtherwise transcribe:\n- Visible text\n- Visual elements/actions\n- Scene descriptions\n- Graphics/text overlays\n- Camera movements\n- Visual transitions\n- Contextual elements matching audio\n\nOutput format:\n<frames>\nFrame [number]: [transcribed content or \"null\"]\n</frames>\n\nAUDIO-VISUAL CORRELATION:\nAfter frame transcription, analyze relationships between visual and audio elements:\n<connections>\n- Match speaker statements to visual cues\n- Link discussed topics to on-screen elements\n- Identify supporting visuals for audio points\n- Note what points the speaker is trying to get at\n- Map visual transitions to topic changes\n- Track visual evidence of described events\n- Connect demonstrations to verbal explanations\n</connections>"
                )

                # Parse and save result
                parsed_response = self._parse_claude_response(response)
                result = {
                    'timestamp': second,
                    'frame_path': str(frame_path),
                    'analysis': parsed_response
                }
                results.append(result)

                # Add delay to avoid rate limits
                time.sleep(1)

            except Exception as e:
                logging.error(f"Error analyzing frame at {second}s: {str(e)}")
                results.append({
                    'timestamp': second,
                    'frame_path': str(frame_path),
                    'analysis': {
                        'frames': 'Analysis failed',
                        'audio_visual_connections': 'Analysis failed'
                    }
                })
                continue

        # Save results for this minute
        if results:
            output_file = self.analysis_dir / f"analysis_minute_{minute}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Saved analysis for minute {minute} to {output_file}")

    def process_directory(self, start_minute=0, end_minute=None):
        """Process all frames in directory within specified minute range"""
        # Find the last frame to determine total duration
        if end_minute is None:
            all_frames = list(self.output_dir.glob("frame_*.jpg"))
            if not all_frames:
                logging.error("No frames found in directory")
                return
            last_frame = max(int(f.stem.split('_')[1]) for f in all_frames)
            end_minute = (last_frame // 60) + 1

        for minute in range(start_minute, end_minute):
            logging.info(f"Processing minute {minute}")
            self.analyze_minute_segment(minute)

def main():
    # Configuration
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Example usage
    output_dir = "output_Systems Engineering Applications Theory Lecture 01-OCT-24 0900-01"
    context_description = "This is a systems engineering lecture focusing on theoretical applications"
    
    analyzer = FrameAnalyzer(api_key, output_dir, context_description)
    analyzer.process_directory(start_minute=0, end_minute=5)  # Process first 5 minutes as example

if __name__ == "__main__":
    main()
