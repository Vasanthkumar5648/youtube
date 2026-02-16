# app.py (UPDATED WITH PDF EXPORT)
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime
import io

# ---------------- Helpers ---------------- #

def extract_video_id(url: str):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_transcript(video_id: str):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except TranscriptsDisabled:
        return "Error: Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "Error: No transcript found for this video."
    except Exception as e:
        return f"Error: {str(e)}"

def chunk_text(text: str, chunk_size: int = 1200):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# ---------------- Model (cached) ---------------- #

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model, device

def summarize_chunk(tokenizer, model, device, chunk: str, max_len=256, min_len=80):
    prompt = (
        "Summarize this transcript chunk in detailed, coherent paragraphs, "
        "preserving all main points and structure:\n\n" + chunk
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)

    summary_ids = model.generate(
        inputs.input_ids,
        max_length=max_len,
        min_length=min_len,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_video(tokenizer, model, device, transcript: str):
    chunks = chunk_text(transcript, chunk_size=1200)
    section_notes = []

    progress = st.progress(0.0, text="Summarizing chunks...")
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        note = summarize_chunk(tokenizer, model, device, chunk)
        section_notes.append(note)
        progress.progress((i + 1) / total, text=f"Summarizing chunk {i + 1}/{total}...")

    combined = " ".join(section_notes)
    global_chunks = chunk_text(combined, chunk_size=2000)
    full_summary_parts = []

    for i, gc in enumerate(global_chunks):
        part = summarize_chunk(tokenizer, model, device, gc, max_len=512, min_len=200)
        full_summary_parts.append(part)

    full_summary = "\n\n".join(full_summary_parts)
    return section_notes, full_summary

# ---------------- PDF Generation ---------------- #

def create_pdf(video_id: str, full_summary: str, section_notes: list):
    """
    Generate a PDF with full summary and section notes
    Returns: BytesIO object containing the PDF
    """
    pdf_buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=1,  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2d5a8c'),
        spaceAfter=10,
        spaceBefore=6,
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=14,
        alignment=4,  # Justify
        spaceAfter=8,
    )
    
    # Add title
    title = Paragraph("YouTube Video Summary Report", title_style)
    story.append(title)
    story.append(Spacer(1, 0.3*inch))
    
    # Add metadata
    metadata_text = f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>Video ID: {video_id}"
    story.append(Paragraph(metadata_text, body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Add divider
    story.append(Paragraph("<hr/>", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Add full summary section
    story.append(Paragraph("Full Video Summary", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Add full summary content
    summary_paragraphs = full_summary.split('\n\n')
    for para in summary_paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))
            story.append(Spacer(1, 0.1*inch))
    
    # Add page break before section notes
    story.append(PageBreak())
    
    # Add section notes
    story.append(Paragraph("Section-by-Section Notes", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    for i, note in enumerate(section_notes, 1):
        section_title = Paragraph(f"<b>Section {i}</b>", heading_style)
        story.append(section_title)
        story.append(Spacer(1, 0.05*inch))
        
        note_para = Paragraph(note.strip(), body_style)
        story.append(note_para)
        story.append(Spacer(1, 0.15*inch))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer

# ---------------- Streamlit UI ---------------- #

def main():
    st.set_page_config(page_title="YouTube Notes & Summary", layout="wide")
    st.title("📺 YouTube Video Detailed Notes & Full Summary")

    st.write("Paste a YouTube URL and get section-wise notes plus a long, full-video summary with PDF export.")

    col1, col2 = st.columns([3, 1])
    
    with col1:
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    
    with col2:
        st.write("")  # Spacing
        generate_btn = st.button("🚀 Generate Notes & Summary", use_container_width=True)

    if generate_btn:
        if not url:
            st.warning("⚠️ Please paste a YouTube URL first.")
            return

        video_id = extract_video_id(url)
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check and try again.")
            return

        # Fetch transcript
        with st.spinner("📥 Fetching transcript..."):
            transcript = get_transcript(video_id)

        if transcript.startswith("Error"):
            st.error(f"❌ {transcript}")
            return

        st.success("✅ Transcript fetched successfully!")
        
        # Load model
        st.info("🤖 Loading summarization model...")
        tokenizer, model, device = load_model()

        # Generate summaries
        with st.spinner("⏳ Generating notes and full summary (this may take a minute)..."):
            section_notes, full_summary = summarize_video(tokenizer, model, device, transcript)

        st.success("✅ Summarization complete!")
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["📋 Full Summary", "📑 Section Notes", "📥 Download"])
        
        with tab1:
            st.subheader("Full Video Summary")
            st.write(full_summary)
        
        with tab2:
            st.subheader("Section-by-Section Notes")
            for i, note in enumerate(section_notes, 1):
                with st.expander(f"📌 Section {i}", expanded=(i==1)):
                    st.write(note)
        
        with tab3:
            st.subheader("Export as PDF")
            st.write("Click the button below to download your complete summary as a PDF file.")
            
            # Generate PDF
            pdf_buffer = create_pdf(video_id, full_summary, section_notes)
            
            # Create download button
            st.download_button(
                label="📄 Download Full Summary as PDF",
                data=pdf_buffer,
                file_name=f"youtube_summary_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            
            st.info("💡 Tip: The PDF includes both the full summary and all section-by-section notes for easy reference.")

