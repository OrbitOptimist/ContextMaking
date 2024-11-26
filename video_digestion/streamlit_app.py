import streamlit as st
import os
from pathlib import Path
import json
from frame_analyzer import FrameAnalyzer
import time
import base64
from PIL import Image

st.set_page_config(page_title="Video Frame Analyzer", layout="wide")

def load_analysis_results(analysis_dir):
    """Load and format analysis results for display"""
    results = []
    analysis_dir = Path(analysis_dir)
    if analysis_dir.exists():
        for file in sorted(analysis_dir.glob("analysis_minute_*.json")):
            with open(file, 'r', encoding='utf-8') as f:
                minute_data = json.load(f)
                results.extend(minute_data)
    return results

def display_frame(frame_path):
    """Display a frame with its analysis"""
    try:
        image = Image.open(frame_path)
        st.image(image, use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

def main():
    st.title("Video Frame Analyzer")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input("Anthropic API Key", type="password")
    if not api_key and "ANTHROPIC_API_KEY" in os.environ:
        api_key = os.environ["ANTHROPIC_API_KEY"]
    
    # Directory input
    output_dir = st.sidebar.text_input(
        "Output Directory Path",
        help="Path to directory containing frames and transcripts"
    )
    
    # Context description
    context_description = st.sidebar.text_area(
        "Context Description",
        help="Provide context about the video content",
        value="This is a lecture video with slides and speaker presentation."
    )
    
    # Minute range selection
    col1, col2 = st.sidebar.columns(2)
    start_minute = col1.number_input("Start Minute", min_value=0, value=0)
    end_minute = col2.number_input("End Minute", min_value=1, value=5)
    
    # Process button
    if st.sidebar.button("Process Frames"):
        if not api_key:
            st.error("Please provide an Anthropic API Key")
            return
        
        if not output_dir:
            st.error("Please provide an output directory")
            return
        
        if not os.path.exists(output_dir):
            st.error("Output directory does not exist")
            return
        
        try:
            analyzer = FrameAnalyzer(api_key, output_dir, context_description)
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process frames
            total_minutes = end_minute - start_minute
            for current_minute in range(start_minute, end_minute):
                status_text.text(f"Processing minute {current_minute}...")
                analyzer.analyze_minute_segment(current_minute)
                progress = (current_minute - start_minute + 1) / total_minutes
                progress_bar.progress(progress)
            
            status_text.text("Processing complete!")
            st.success("Frame analysis completed successfully!")
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
    
    # Results viewer
    st.header("Analysis Results")
    
    if output_dir and os.path.exists(output_dir):
        analysis_dir = Path(output_dir) / "frame_analysis"
        if analysis_dir.exists():
            results = load_analysis_results(analysis_dir)
            
            if results:
                # Create tabs for each minute
                minute_tabs = {}
                for result in results:
                    minute = result['timestamp'] // 60
                    if minute not in minute_tabs:
                        minute_tabs[minute] = []
                    minute_tabs[minute].append(result)
                
                # Display tabs
                tabs = st.tabs([f"Minute {minute}" for minute in sorted(minute_tabs.keys())])
                
                for tab_idx, minute in enumerate(sorted(minute_tabs.keys())):
                    with tabs[tab_idx]:
                        for result in minute_tabs[minute]:
                            st.subheader(f"Frame at {result['timestamp']}s")
                            
                            # Display frame and analysis side by side
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                display_frame(result['frame_path'])
                            with col2:
                                st.markdown("### Analysis")
                                st.write(result['analysis'])
                            
                            st.divider()
            else:
                st.info("No analysis results found. Process some frames to see results here.")
        else:
            st.info("No analysis results found. Process some frames to see results here.")

if __name__ == "__main__":
    main()
