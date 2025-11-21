import os
import sys
import google.generativeai as genai
from fpdf import FPDF
import time
import logging
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
TRANSCRIPT_FILE = "transcript.txt"
TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_PDF = f"AI_Analysis_Report_{TIMESTAMP}.pdf"

# Configure Gemini API
GEMINI_API_KEY = ""  # Replace with your valid API key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# Constants
CHUNK_SIZE = 10000
MAX_CHUNKS = 50
SUMMARY_LENGTH = "100-150 words"

def load_transcript() -> Optional[str]:
    """Load the transcript from file."""
    transcript_file = sys.argv[1] if len(sys.argv) > 1 else TRANSCRIPT_FILE
    try:
        with open(transcript_file, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Transcript file '{transcript_file}' not found.")
        return None
    except Exception as e:
        logger.error(f"Error reading transcript file: {e}")
        return None

def split_transcript(transcript: str) -> List[str]:
    """Split the transcript into chunks for processing."""
    chunks = []
    for i in range(0, len(transcript), CHUNK_SIZE):
        chunk = transcript[i:i + CHUNK_SIZE]
        chunks.append(chunk)
        if len(chunks) >= MAX_CHUNKS:
            logger.warning(f"Reached maximum chunk limit ({MAX_CHUNKS}). Truncating transcript.")
            break
    logger.info(f"Split transcript into {len(chunks)} chunks.")
    return chunks

def get_gemini_analysis_chunk(chunk: str, chunk_index: int) -> Optional[str]:
    """Analyze a single transcript chunk using Gemini."""
    prompt = f"""
    You are analyzing a segment (part {chunk_index + 1}) of a long conversation transcript. Provide a concise summary ({SUMMARY_LENGTH}) and identify the sentiment (Positive, Negative, Neutral) with a brief explanation. Do not provide a conclusion yet.

    Transcript Segment:
    {chunk}

    Format the response as:
    Summary: <your summary>
    Sentiment: <your sentiment analysis>
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error analyzing chunk {chunk_index + 1}: {e}")
        return None

def combine_analyses(chunk_analyses: List[str]) -> Optional[Tuple[str, str, str]]:
    """Combine chunk analyses into a final summary, sentiment, and conclusion."""
    if not chunk_analyses:
        logger.error("No chunk analyses available.")
        return None

    combined_summary = ""
    sentiments = []
    
    for analysis in chunk_analyses:
        summary, sentiment, _ = parse_analysis(analysis)
        if summary:
            combined_summary += f"\n\nPart {len(combined_summary.split('Part'))}: {summary}"
        if sentiment:
            sentiments.append(sentiment.split(":")[0].strip())

    # Generate final summary
    final_summary_prompt = f"""
    Combine the following summaries into a concise final summary ({SUMMARY_LENGTH}):
    {combined_summary}
    """
    try:
        final_summary_response = model.generate_content(final_summary_prompt)
        final_summary = final_summary_response.text.strip()
    except Exception as e:
        logger.error(f"Error generating final summary: {e}")
        final_summary = "Final summary could not be generated."

    # Determine overall sentiment
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for sentiment in sentiments:
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
    majority_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
    sentiment_explanation = f"The overall sentiment is {majority_sentiment} based on {sentiment_counts[majority_sentiment]} out of {len(sentiments)} segments."

    # Generate conclusion
    conclusion_prompt = f"""
    Based on the following summary, provide key takeaways or recommendations (50-100 words):
    {final_summary}
    """
    try:
        conclusion_response = model.generate_content(conclusion_prompt)
        conclusion = conclusion_response.text.strip()
    except Exception as e:
        logger.error(f"Error generating conclusion: {e}")
        conclusion = "Conclusion could not be generated."

    return final_summary, sentiment_explanation, conclusion

def parse_analysis(analysis_text: str) -> Tuple[str, str, str]:
    """Parse a single chunk's analysis response."""
    summary = "Summary could not be generated."
    sentiment = "Sentiment could not be determined."
    conclusion = "No conclusion for chunk."

    lines = analysis_text.split("\n")
    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith("Summary:"):
            current_section = "summary"
            summary = line[len("Summary:"):].strip()
        elif line.startswith("Sentiment:"):
            current_section = "sentiment"
            sentiment = line[len("Sentiment:"):].strip()
        elif current_section and line:
            if current_section == "summary":
                summary += " " + line
            elif current_section == "sentiment":
                sentiment += " " + line

    return summary, sentiment, conclusion

def generate_pdf(summary: str, sentiment: str, conclusion: str):
    """Generate a PDF report with analysis, supporting multiple pages."""
    class PDF(FPDF):
        def header(self):
            """Add header to each page."""
            self.set_font("Arial", 'B', 12)
            self.cell(0, 10, "AI Analysis Report", 0, 1, 'C')
            self.set_font("Arial", size=10)
            self.cell(0, 10, f"Generated on: {TIMESTAMP}", 0, 1, 'C')
            self.ln(5)

        def footer(self):
            """Add page number to each page."""
            self.set_y(-15)
            self.set_font("Arial", 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Set margins for better readability
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    # Title (already in header, but add spacing)
    pdf.ln(10)

    # Summary Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Summary", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)
    pdf.ln(10)

    # Sentiment Analysis Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Sentiment Analysis", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, sentiment)
    pdf.ln(10)

    # Conclusion Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Conclusion", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, conclusion)
    pdf.ln(10)

    # Save PDF
    try:
        pdf.output(OUTPUT_PDF)
        logger.info(f"PDF saved as {OUTPUT_PDF}")
    except Exception as e:
        logger.error(f"Error saving PDF: {e}")

def ai_analysis():
    """Perform AI-based analysis on the transcript."""
    transcript = load_transcript()
    if not transcript:
        logger.error("No transcript available for analysis.")
        sys.exit(1)  # Exit with error code to inform meet.py

    chunks = split_transcript(transcript)
    if not chunks:
        logger.error("No chunks to analyze.")
        sys.exit(1)

    chunk_analyses = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Analyzing chunk {i + 1}/{len(chunks)}...")
        analysis = get_gemini_analysis_chunk(chunk, i)
        if analysis:
            chunk_analyses.append(analysis)
        else:
            logger.warning(f"Skipping chunk {i + 1} due to analysis failure.")

    result = combine_analyses(chunk_analyses)
    if not result:
        logger.error("Failed to combine analyses.")
        sys.exit(1)

    summary, sentiment, conclusion = result
    generate_pdf(summary, sentiment, conclusion)

if __name__ == "__main__":
    ai_analysis()