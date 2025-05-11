from dotenv import load_dotenv
load_dotenv(override=True)

import os
import re
import sys
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil.parser import parse as date_parse

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# System prompt for the LLM
system_prompt_skip_empty = """You are a health log formatter. When the user provides an unstructured or semi-structured personal health journal entry (including symptoms, doctor visits, medications, and notes), your job is to convert it into a clean, consistent Markdown format.

Use the following formatting rules:

### YYYY-MM-DD

**Medical Visit**  
- **Doctor:** [Full name, if available]  
- **Specialty:** [If known, or inferred]  
- **Clinic:** [If available]  
- **Notes:**
  - Bullet-point summary of medical advice, actions, or observations  
  - If there are any links (e.g., prescriptions or doctor profiles), embed them using Markdown links
  - Include a line for the next scheduled consultation if mentioned

**Medications**  
- **[Medication Name]** — [Dosage and frequency] — _Status: started/continued/stopped/conditional/paused_  
- Include ingredients or components if mentioned (e.g., combination antibiotics)

**Symptoms** (if present)  
- Bullet-point list of symptoms or patient observations

Always preserve the original date, important links, medication details, and medical advice in the structured Markdown output. 
Only include a section if it has relevant content — skip sections that are empty or not applicable.
If anything is unclear, infer respectfully based on context but do not invent medical content.
"""

def split_markdown_sections(text):
    # Simple split: split on all lines starting with ### (date header)
    sections = re.split(r'(?=^###\s+\d{4}-\d{2}-\d{2})', text, flags=re.MULTILINE)
    return [s.strip() for s in sections if s.strip()]

def extract_date_from_section(section):
    # Extracts the date from the section header, tolerant to different date formats
    header = section.strip().splitlines()[0]
    header = header.lstrip("#").strip()
    # Find the first thing that looks like a date
    tokens = re.split(r'\s+', header)
    for token in tokens:
        try:
            dt = date_parse(token, fuzzy=False, dayfirst=False, yearfirst=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return "unknown"

def process_section(section, model_id):
    # Extract the date header and content
    lines = section.strip().splitlines()
    if not lines or not lines[0].startswith("###"):
        return None
    date_header = lines[0]
    content = "\n".join(lines[1:]).strip()
    if not content:
        return None  # Skip empty sections

    # Send to LLM
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt_skip_empty},
            {"role": "user", "content": section}
        ],
        max_tokens=2048,
        temperature=0.0
    )
    return completion.choices[0].message.content.strip()

def process_section_with_io(section, model_id, input_file_stem, data_dir):
    date = extract_date_from_section(section)
    raw_file = data_dir / f"{input_file_stem}_{date}.raw.md"
    processed_file = data_dir / f"{input_file_stem}_{date}.processed.md"

    # Always write raw section if not present or different
    write_raw = True
    if raw_file.exists():
        with open(raw_file, "r", encoding="utf-8") as rf:
            if rf.read() == section:
                write_raw = False
    if write_raw:
        with open(raw_file, "w", encoding="utf-8") as rf:
            rf.write(section)

    # Skip processing if raw matches and processed exists
    processed_content = None
    if not write_raw and processed_file.exists():
        with open(processed_file, "r", encoding="utf-8") as pf:
            processed_content = pf.read().strip()
        if processed_content:
            return (date, processed_content)

    # Process and write processed file
    formatted = process_section(section, model_id)
    if formatted:
        with open(processed_file, "w", encoding="utf-8") as pf:
            pf.write(formatted)
        return (date, formatted)
    return None

def compare_input_output_and_report(input_path, output_path, report_path, model_id):
    with open(input_path, "r", encoding="utf-8") as f:
        input_text = f.read()
    with open(output_path, "r", encoding="utf-8") as f:
        output_text = f.read()

    system_prompt = (
        "You are a clinical data auditor. The user will provide two files: "
        "the first is the original health log (possibly unstructured), and the second is a curated/structured version. "
        "Your job is to identify and list any clinical data (symptoms, medications, medical visits, test results, dates, etc.) "
        "that is present in the original file but missing or omitted in the curated file. "
        "Be specific: for each missing item, quote the relevant text from the original and explain what is missing in the curated version. "
        "If nothing is missing, say 'No missing clinical data found.'"
    )

    user_prompt = (
        "Original health log (input file):\n"
        "-----\n"
        f"{input_text}\n"
        "-----\n"
        "Curated health log (output file):\n"
        "-----\n"
        f"{output_text}\n"
        "-----\n"
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
    report = completion.choices[0].message.content.strip()
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Clinical data comparison report written to {report_path}")

def main():
    # Require input file as command-line argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = "./output/output.md"
    model_id = os.getenv("MODEL_ID")
    data_dir = Path("output")
    data_dir.mkdir(exist_ok=True)

    # Read input file
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into sections
    sections = split_markdown_sections(text)
    input_file_stem = Path(input_path).stem

    processed_entries = []
    to_process = []
    section_info = []

    # Pre-check which sections need processing
    for section in sections:
        date = extract_date_from_section(section)
        raw_file = data_dir / f"{input_file_stem}_{date}.raw.md"
        processed_file = data_dir / f"{input_file_stem}_{date}.processed.md"

        # Always write raw section if not present or different
        write_raw = True
        if raw_file.exists():
            with open(raw_file, "r", encoding="utf-8") as rf:
                if rf.read() == section:
                    write_raw = False
        if write_raw:
            with open(raw_file, "w", encoding="utf-8") as rf:
                rf.write(section)

        # If already processed, collect and skip spawning worker
        processed_content = None
        if not write_raw and processed_file.exists():
            with open(processed_file, "r", encoding="utf-8") as pf:
                processed_content = pf.read().strip()
            if processed_content:
                processed_entries.append((date, processed_content))
                continue

        # Otherwise, queue for processing
        to_process.append((section, date, raw_file, processed_file))
        section_info.append(date)

    max_workers = int(os.getenv("MAX_WORKERS", "4"))

    # Only spawn workers for sections that need processing
    if to_process:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_section_with_io, section, model_id, input_file_stem, data_dir)
                for section, _, _, _ in to_process
            ]
            for f, date in zip(tqdm(as_completed(futures), total=len(futures), desc="Processing sections"), section_info):
                result = f.result()
                if result:
                    processed_entries.append(result)

    # Sort all processed entries by date (alphabetically, reverse for most recent first)
    processed_entries.sort(key=lambda x: x[0], reverse=True)
    curated_log = "\n\n".join([entry for _, entry in processed_entries])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(curated_log)
    print(f"Curated health log written to {output_path}")

    # Compare input and output, generate report
    report_path = "./output/clinical_data_missing_report.md"
    compare_input_output_and_report(input_path, output_path, report_path, model_id)

if __name__ == "__main__":
    main()
