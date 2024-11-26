# Video Frame Analyzer

This tool analyzes video frames and transcripts using Claude's vision capabilities to create detailed frame-by-frame analysis with audio-visual correlations.

## Prerequisites

- Python 3.8+
- Anthropic API key
- Processed video output directory containing:
  - Frames (frame_X.jpg)
  - Transcripts (transcript_X.txt)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up your Anthropic API key:
```bash
# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="your-api-key"
```

## Usage

### Command Line Processing

```python
from frame_analyzer import FrameAnalyzer

analyzer = FrameAnalyzer(
    api_key="your-api-key",
    output_dir="path/to/output_directory",
    context_description="Brief description of video context"
)

# Process specific minute range
analyzer.process_directory(start_minute=0, end_minute=5)
```

### Streamlit Interface

Run the Streamlit app for an interactive interface:

```bash
streamlit run streamlit_app.py
```

The Streamlit interface provides:
- Easy directory and API key configuration
- Context description input
- Minute range selection
- Progress tracking
- Results viewer with frame-by-frame analysis

## Output Format

The analyzer creates a `frame_analysis` directory within your output directory containing JSON files for each processed minute:

```
output_directory/
├── frame_analysis/
│   ├── analysis_minute_0.json
│   ├── analysis_minute_1.json
│   └── ...
```

Each JSON file contains:
- Frame timestamp
- Frame path
- Claude's analysis including:
  - Frame content transcription
  - Audio-visual correlations
  - Context relationships

## Tips

1. Process in smaller minute chunks for better manageability
2. Provide detailed context descriptions for better analysis
3. Use the Streamlit interface for easy visualization of results
4. Monitor the progress bar and logs for processing status
5. Check the frame_analysis directory for detailed JSON outputs

## Example Context Description

Good context description example:
```python
context_description = """
This is a technical lecture on systems engineering covering theoretical applications.
The speaker uses slides and diagrams to explain concepts.
Key topics include system design principles and methodologies.
"""
