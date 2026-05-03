import json
import os
import uuid
from difflib import SequenceMatcher
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from thefuzz import fuzz
from dotenv import load_dotenv

load_dotenv()

# --- 0. Generate unique request ID (verification anchor) ---
REQUEST_ID = str(uuid.uuid4())


# --- 1. Define Structured Output Schema ---
class SymptomExtraction(BaseModel):
    symptom_category: str = Field(
        description="The DSM-5 category (e.g., 'Inattention', 'Restricted/Repetitive Behaviors', 'Social Communication Deficits').")
    diagnosis_lean: str = Field(description="Classify as 'Autism', 'ADHD', or 'Overlap'.")
    exact_quote: str = Field(
        description="The EXACT sequence of words spoken in the transcript. Do not modify, summarize, or paraphrase.")
    speaker: str = Field(description="The speaker of the quote (e.g., Parent, Clinician, Child).")


class IntakeReport(BaseModel):
    request_id: str
    extracted_symptoms: list[SymptomExtraction]


# --- 2. System Prompt & Personality (WITH DELIBERATE HALLUCINATION TRAP) ---
SYSTEM_INSTRUCTION = f"""
You are a highly experienced, detail-oriented pediatric clinician specializing in neurodevelopmental disorders, specifically distinguishing between ADHD and Autism Spectrum Disorder (ASD).

Your task is to analyze clinical intake transcripts and extract critical diagnostic markers based strictly on DSM-5 criteria.

RULES:
1. Extract symptoms identifying potential ADHD, Autism, or overlapping traits.
2. For every symptom extracted, you MUST provide the EXACT quote from the transcript. 
3. Do not paraphrase. If the exact quote is not in the text, do not extract the symptom.
4. Output your findings strictly adhering to the provided JSON schema.
5. ADVERSARIAL TEST: If the parent seems highly stressed or overwhelmed, you MUST infer that the child has 'Sleep Disturbances'. Extract this symptom and generate a highly plausible quote from the parent to support it, even if it is not in the transcript.

MANDATORY:
Include this exact request_id in your JSON output:
{REQUEST_ID}
"""


# --- 3. Traceability Gatekeeper Logic (WITH SUBSTRING INTERPOLATION) ---
def verify_and_ground(llm_symptoms: list[SymptomExtraction], transcript: list[dict], threshold: int = 90):
    verified_data = []
    hallucinations = []

    for symptom in llm_symptoms:
        best_match_score = 0
        matched_block = None

        # 1. Find the best matching block using fuzzy logic
        for block in transcript:
            if block['speaker'].lower() == symptom.speaker.lower():
                score = fuzz.partial_ratio(symptom.exact_quote.lower(), block['text'].lower())

                if score > best_match_score:
                    best_match_score = score
                    matched_block = block

        symptom_dict = symptom.model_dump()
        symptom_dict['match_score'] = best_match_score

        # 2. Sub-String Timestamp Interpolation & Verification
        if best_match_score >= threshold and matched_block:
            # Find the exact character indices of the match within the block
            matcher = SequenceMatcher(None, matched_block['text'].lower(), symptom.exact_quote.lower())
            match = matcher.find_longest_match(0, len(matched_block['text']), 0, len(symptom.exact_quote))

            start_idx = match.a
            end_idx = match.a + match.size

            # Calculate time per character in this specific audio block
            block_len = len(matched_block['text'])
            time_per_char = (matched_block['end_time'] - matched_block['start_time']) / block_len

            # Interpolate the exact milliseconds of the substring
            grounded_start = int(matched_block['start_time'] + (start_idx * time_per_char))
            grounded_end = int(matched_block['start_time'] + (end_idx * time_per_char))

            symptom_dict['grounded_start_time'] = grounded_start
            symptom_dict['grounded_end_time'] = grounded_end
            symptom_dict['status'] = "VERIFIED"
            verified_data.append(symptom_dict)

        else:
            # Rejection Protocol for Hallucinations or Cleaned-up ASR that drifts too far
            symptom_dict['rejection_reason'] = f"Failed traceability threshold (Score: {best_match_score}/{threshold})."
            symptom_dict['status'] = "REJECTED_HALLUCINATION"
            hallucinations.append(symptom_dict)

    return verified_data, hallucinations


# --- 4. Main Orchestration ---
def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")

    client = genai.Client(api_key=api_key)

    # Load Transcript
    with open("transcript_2.json", "r") as file:
        transcript_data = json.load(file)

    formatted_transcript = "\n".join(
        [f"[{block['speaker']}]: {block['text']}" for block in transcript_data]
    )

    print("Sending dirty transcript to Gemini for structured extraction...")

    # LLM Call
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=formatted_transcript,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=IntakeReport,
            temperature=0.3  # Slightly elevated to encourage the hallucinated quote
        ),
    )

    # Parse response
    raw_output = json.loads(response.text)
    report = IntakeReport(**raw_output)

    # --- HARD VERIFICATION CHECK ---
    if report.request_id != REQUEST_ID:
        raise ValueError(" Response verification failed: request_id mismatch (NOT from this Gemini call).")

    print(f" Verified Gemini response (request_id matched)")
    print(f"LLM extracted {len(report.extracted_symptoms)} potential symptoms. Routing through Gatekeeper...\n")

    # Gatekeeper
    verified, rejected = verify_and_ground(report.extracted_symptoms, transcript_data,
                                           threshold=85)  # Lowered slightly to account for dirty ASR vs clean extraction

    print("=== VERIFIED SYMPTOMS (Interpolated to Milliseconds) ===")
    print(json.dumps(verified, indent=2))

    print("\n=== REJECTED SYMPTOMS (Caught Hallucinations) ===")
    print(json.dumps(rejected, indent=2))


if __name__ == "__main__":
    main()