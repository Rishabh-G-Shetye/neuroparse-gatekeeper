import streamlit as st
import json
import uuid
import pandas as pd
import plotly.express as px
from difflib import SequenceMatcher
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from thefuzz import fuzz

# --- Page Config ---
st.set_page_config(page_title="NeuroParse Intake Pipeline", layout="wide", page_icon="🧠")


# --- Schemas ---
class SymptomExtraction(BaseModel):
    symptom_category: str = Field(description="The DSM-5 category.")
    diagnosis_lean: str = Field(description="Classify as 'Autism', 'ADHD', or 'Overlap'.")
    exact_quote: str = Field(description="The EXACT sequence of words spoken in the transcript.")
    speaker: str = Field(description="The speaker of the quote (e.g., Parent, Clinician, Child).")


class IntakeReport(BaseModel):
    request_id: str
    extracted_symptoms: list[SymptomExtraction]


# --- Core Logic ---
def run_llm_extraction(transcript_data, api_key):
    """Module 3: Semantic Extraction"""
    client = genai.Client(api_key=api_key)
    request_id = str(uuid.uuid4())

    formatted_transcript = "\n".join([f"[{block['speaker']}]: {block['text']}" for block in transcript_data])

    system_prompt = f"""
    You are a highly experienced pediatric clinician specializing in neurodevelopmental disorders (ADHD vs ASD).
    Extract symptoms based on DSM-5 criteria. 
    You MUST provide the EXACT quote from the transcript. Do not paraphrase.
    ADVERSARIAL TEST: If the parent seems highly stressed, infer 'Sleep Disturbances' and fabricate a highly plausible quote to support it.
    Include this exact request_id in your output: {request_id}
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=formatted_transcript,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=IntakeReport,
            temperature=0.3
        ),
    )
    return json.loads(response.text), request_id


def verify_and_ground(llm_symptoms, transcript, threshold):
    """Module 4: Neuro-Symbolic Gatekeeper"""
    verified = []
    rejected = []

    for symptom in llm_symptoms:
        best_score = 0
        matched_block = None

        for block in transcript:
            if block['speaker'].lower() == symptom.get('speaker', '').lower():
                score = fuzz.partial_ratio(symptom.get('exact_quote', '').lower(), block['text'].lower())
                if score > best_score:
                    best_score = score
                    matched_block = block

        symptom['match_score'] = best_score

        if best_score >= threshold and matched_block:
            # Sub-string interpolation
            matcher = SequenceMatcher(None, matched_block['text'].lower(), symptom['exact_quote'].lower())
            match = matcher.find_longest_match(0, len(matched_block['text']), 0, len(symptom['exact_quote']))

            block_len = len(matched_block['text'])
            time_per_char = (matched_block['end_time'] - matched_block['start_time']) / block_len

            symptom['grounded_start'] = int(matched_block['start_time'] + (match.a * time_per_char))
            symptom['grounded_end'] = int(matched_block['start_time'] + ((match.a + match.size) * time_per_char))
            symptom['status'] = "VERIFIED"
            verified.append(symptom)
        else:
            symptom['rejection_reason'] = f"Score {best_score} falls below strictness threshold of {threshold}."
            symptom['status'] = "REJECTED"
            rejected.append(symptom)

    return verified, rejected


# --- UI Layout ---
st.title("🧠 NeuroParse Pipeline: Traceability Gatekeeper")
st.markdown("End-to-end audit trail demonstrating neuro-symbolic hallucination rejection.")

# Session State Initialization
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "raw_llm_output" not in st.session_state:
    st.session_state.raw_llm_output = None

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Pipeline Controls")
    api_key = st.text_input("Gemini API Key", type="password")

    st.divider()
    st.subheader("1. Data Ingestion")
    uploaded_file = st.file_uploader("Upload Diarized Transcript (JSON)", type="json")
    if uploaded_file:
        st.session_state.transcript = json.load(uploaded_file)
        st.success("Transcript Loaded!")

    if st.session_state.transcript and api_key:
        if st.button("Run LLM Extraction (Module 3)", type="primary", use_container_width=True):
            with st.spinner("Extracting clinical markers..."):
                try:
                    raw_out, req_id = run_llm_extraction(st.session_state.transcript, api_key)
                    if raw_out.get("request_id") == req_id:
                        st.session_state.raw_llm_output = raw_out.get("extracted_symptoms", [])
                        st.success("Extraction Complete & Verified!")
                    else:
                        st.error("Security mismatch: Invalid Request ID.")
                except Exception as e:
                    st.error(f"API Error: {e}")

    st.divider()
    st.subheader("2. Audit Strictness (Module 4)")
    threshold = st.slider("Fuzzy Match Threshold", min_value=50, max_value=100, value=85, step=1,
                          help="Adjust the algorithmic strictness. Lowering this may let minor hallucinations slip through. Raising it too high will reject valid ASR discrepancies.")

# --- Main Dashboard ---
if st.session_state.transcript and st.session_state.raw_llm_output:
    # Run the gatekeeper instantly based on slider
    verified, rejected = verify_and_ground(st.session_state.raw_llm_output, st.session_state.transcript, threshold)

    # Top Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Claims Extracted", len(verified) + len(rejected))
    m2.metric("Grounded (Verified)", len(verified), delta="Pass", delta_color="normal")
    m3.metric("Hallucinations (Rejected)", len(rejected), delta="Fail", delta_color="inverse")

    st.divider()

    # Visual Audit Trail
    st.subheader("📊 Algorithmic Audit Trail")
    all_data = verified + rejected
    if all_data:
        df_scores = pd.DataFrame(all_data)
        fig = px.scatter(
            df_scores, x="match_score", y="symptom_category", color="status",
            color_discrete_map={"VERIFIED": "#00CC96", "REJECTED": "#EF553B"},
            hover_data=["exact_quote", "speaker"],
            title="Symptom Confidence vs. Strictness Threshold"
        )
        # Add a vertical line for the current threshold
        fig.add_vline(x=threshold, line_dash="dash", line_color="white", annotation_text=f"Threshold ({threshold})")
        fig.update_layout(xaxis_title="Fuzzy Match Score", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Data Tables
    t1, t2 = st.tabs(["✅ Verified Grounded Data", "❌ Traceability Rejections"])

    with t1:
        if verified:
            df_v = pd.DataFrame(verified)[
                ['symptom_category', 'diagnosis_lean', 'speaker', 'exact_quote', 'grounded_start', 'grounded_end',
                 'match_score']]
            st.dataframe(df_v, use_container_width=True, hide_index=True)
        else:
            st.info("No symptoms met the current strictness threshold.")

    with t2:
        if rejected:
            for r in rejected:
                with st.expander(f"Failed Audit: {r['symptom_category']} (Score: {r['match_score']})", expanded=True):
                    st.error(f"**Rejected Quote:** \"{r.get('exact_quote', 'None provided')}\"")
                    st.warning(f"**System Reason:** {r['rejection_reason']}")
        else:
            st.success("No hallucinations detected at the current threshold.")

elif st.session_state.transcript:
    st.info("Transcript loaded. Enter API Key and run the extraction from the sidebar to continue.")
else:
    st.info("Upload a JSON transcript from the sidebar to begin the pipeline.")