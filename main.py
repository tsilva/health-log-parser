from dotenv import load_dotenv; load_dotenv(override=True)
import os, re, sys
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil.parser import parse as date_parse
import argparse
import hashlib

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# LLM system prompt for health log formatting
system_prompt_skip_empty = """
You are a health log formatter and extractor. Your task is to convert each unstructured or semi-structured personal health journal entry into a structured Markdown block, one per date, capturing all relevant clinical details. Do not omit any clinical data (symptoms, medications, visits, test results, dates, etc.) present in the input.

Instructions:
- For each entry, output a Markdown section starting with '#### YYYY-MM-DD' (use the date from the entry).
- Under each date, use bullet points to list all clinical events, tests, symptoms, medications, diagnoses, and notes.
- For lab tests, include test name, value, reference range, and interpretation if available.
- For doctor visits, include doctor name, location, prescriptions (with dose, frequency, duration), diagnoses, and advice.
- For symptoms, list them with date and any relevant context.
- For appointments, specify date and purpose.
- If a web link is present, format as [description](url).
- If information is missing or unclear, include it as a note.
- Do not invent or omit information; only use what is present in the input.
- Preserve all clinical details, even if they seem minor.
- If an entry does not mention symptoms, diagnosis, or additional clinical details, do NOT add a note such as "No symptoms, diagnosis, or additional clinical details provided in the entry." Only include information actually present in the input.
- Regardless of input language, all output should be in English.

SAMPLE OUTPUT 1:

#### 2023-04-11

- Colonoscopy performed by **Dr. Jones (Gastroenterologist)** at **City Hospital**
    - Findings:
        - No polyps found, normal mucosa.
        - Biopsy taken for further analysis.
    - Notes:
        - Follow-up in 2 weeks for biopsy results.
        - Preparation was difficult, but manageable.

SAMPLE OUTPUT 2:

### 2023-04-12

- [Lab testing at LabABC](https://lababc.com/test/12345)
    - Values:
        - **Hemoglobin:** 13.2 g/dL (ref: 12-16, normal)
        - **Leukocytes:** 5.1 x10^9/L (ref: 4-10, normal)
        - **Ferritin:** 8 ng/mL (ref: 15-150, low)
    - Notes:
        - Low ferritin indicates possible iron deficiency.

- Doctor visit with **Dr. Smith (Gastroenterologist)** at **City Hospital**
    - Prescription:
        - **Iron Protein Succinylate 100 mg**, 1 tablet daily for 3 month
        - **Vitamin C 500 mg**, 1 tablet daily for 3 month
        - **Folic Acid 1 mg**, 1 tablet daily for 3 month
        - **Vitamin B12 1000 mcg**, 1 tablet weekly for 3 month
    - Diagnosis:
        - Iron deficiency anemia
    - Notes:
        - Advised dietary changes to include more iron-rich foods.
        - Recommended follow-up in 3 months.

- I did not feel well on 2023-04-10, had a headache and fatigue.
- I have a follow-up appointment scheduled for 2023-05-01.
"""

# Extract date from section header, tolerant to different formats
def extract_date_from_section(section):
    header = section.strip().splitlines()[0].lstrip("#").strip()
    # Normalize dashes to standard hyphen-minus
    header = header.replace("–", "-").replace("—", "-")
    for token in re.split(r'\s+', header):
        return date_parse(token, fuzzy=False, dayfirst=False, yearfirst=True).strftime("%Y-%m-%d")
    
def get_short_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

def get_file_short_hash(filepath):
    with open(filepath, "r", encoding="utf-8") as f: return get_short_hash(f.read())

def process(input_path):
    model_id = os.getenv("MODEL_ID")

    # Create output path
    short_hash = get_file_short_hash(input_path)
    data_dir = Path("output") / short_hash
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "output.md"

    # Run LLM to process a section, return formatted markdown
    def _process(raw_section):
        # Write raw section to file
        date = extract_date_from_section(raw_section)
        raw_file = data_dir / f"{date}.raw.md"
        raw_file.write_text(raw_section, encoding="utf-8")

        # The existence/up-to-date check is now outside
        for _ in range(3):
            # Run LLM to process the section
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt_skip_empty},
                    {"role": "user", "content": raw_section}
                ],
                max_tokens=2048,
                temperature=0.0
            )
            processed_section = completion.choices[0].message.content.strip()

            # Run LLM to validate the processed section
            system_prompt = (
                "You are a clinical data auditor. The user will provide two files: "
                "the first is the original health log (possibly unstructured), and the second is a curated/structured version. "
                "Your job is to identify and list any clinical data (symptoms, medications, medical visits, test results, dates, etc.) "
                "that is present in the original file but missing or omitted in the curated file. "
                "Be specific: for each missing item, quote the relevant text from the original and explain what is missing in the curated version. "
                "If nothing is missing, reply only with '$OK$' (without quotes) and do not add any other text."
                "If you find any error in the curated version, after describing all issues, output '$FAILED$."
            )
            user_prompt = (
                "Original health log (input section):\n-----\n"
                f"{raw_section}\n-----\n"
                "Curated health log (output section):\n-----\n"
                f"{processed_section}\n-----\n"
                "Please list any clinical data present in the original but missing in the curated version."
            )
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2048,
                temperature=0.0
            )
            
            # If the validation does not return "$OK$", retry processing
            error_content = completion.choices[0].message.content.strip()
            if "$OK$" not in error_content: continue

            # If validation passes, write the processed section to file
            raw_hash = get_short_hash(raw_section) 
            processed_file = data_dir / f"{date}.processed.md"
            processed_file.write_text(f"{raw_hash}\n{processed_section}", encoding="utf-8")
            print(f"Processed section for date {date} written to {processed_file}")
            
            # Return True to indicate successful processing
            return True
        
        # If all retries failed, return False
        return False 

    # Read and split input file into sections
    with open(input_path, "r", encoding="utf-8") as f: input_text = f.read()
    sections = [s.strip() for s in re.split(r'(?=^###)', input_text, flags=re.MULTILINE) if s.strip()]

    # Assert that each section contains exactly one '###'
    for section in sections:
        count = section.count('###')
        assert count == 1, f"Section does not contain exactly one ###:\n{section}"

    # Assert no duplicate dates in sections
    dates = [extract_date_from_section(section) for section in sections]
    # Print duplicate dates if any
    if len(dates) != len(set(dates)):
        duplicates = {date for date in dates if dates.count(date) > 1}
        print(f"Duplicate dates found: {duplicates}")
        sys.exit(1)
    
    # Precompute which sections need processing
    to_process = []
    for section in sections:
        date = extract_date_from_section(section)
        raw_hash = get_short_hash(section)
        processed_file = data_dir / f"{date}.processed.md"
        if processed_file.exists():
            processed_text = processed_file.read_text(encoding="utf-8").strip()
            _raw_hash = processed_text.splitlines()[0].strip()
            if _raw_hash == raw_hash: continue 
        to_process.append(section)

    # Process sections in parallel (only those that need processing)
    max_workers = int(os.getenv("MAX_WORKERS", "4"))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process, section) for section in to_process]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing sections"):
            _.result()

    # Write the final curated health log
    processed_files = list(data_dir.glob("*.processed.md"))
    processed_files = sorted(processed_files, key=lambda f: f.stem, reverse=True)
    processed_entries = ["\n".join(f.read_text(encoding="utf-8").splitlines()[1:]) for f in processed_files]
    processed_text = "\n\n".join(processed_entries)
    with open(output_path, "w", encoding="utf-8") as f: f.write(processed_text)
    print(f"Curated health log written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Health log parser and validator")
    parser.add_argument("input_file", help="Original input file")
    
    args = parser.parse_args()
    process(args.input_file)

if __name__ == "__main__":
    main()
