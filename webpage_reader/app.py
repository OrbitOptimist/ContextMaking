import streamlit as st
from webpage_processor import WebpageProcessor

def main():
    st.title("Webpage Reader")
    st.write("Enter a URL to extract readable content and convert it to markdown format.")
    
    # Initialize the processor
    if 'processor' not in st.session_state:
        st.session_state.processor = WebpageProcessor()
    
    # URL input
    url = st.text_input("Enter URL:", "")
    
    if st.button("Process Webpage"):
        if url:
            with st.spinner("Processing webpage..."):
                try:
                    # Reset processor state
                    st.session_state.processor.reset()
                    
                    # Get content
                    markdown_content = st.session_state.processor.get_readable_content(url)
                    
                    if markdown_content:
                        # Display the markdown content
                        st.markdown(markdown_content)
                        
                        # Add download button for markdown content
                        st.download_button(
                            label="Download Markdown",
                            data=markdown_content,
                            file_name="webpage_content.md",
                            mime="text/markdown"
                        )
                    else:
                        st.error("No content could be extracted from the webpage.")
                except Exception as e:
                    st.error(f"Error processing webpage: {str(e)}")
        else:
            st.warning("Please enter a URL.")

if __name__ == "__main__":
    main()
