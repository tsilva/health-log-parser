from dotenv import load_dotenv
load_dotenv(override=True)

import os
import re
import sys
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    # Improved: splits on lines starting with ### followed by a date, capturing the header and all following lines
    pattern = r"(^###\s+\d{4}-\d{2}-\d{2}.*(?:\n(?!### ).*)*)"
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return [m.strip() for m in matches if m.strip()]

def extract_date_from_section(section):
    # Extracts the date from the section header
    m = re.match(r"^###\s+(\d{4}-\d{2}-\d{2})", section.strip())
    return m.group(1) if m else "unknown"

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

def main():
    # Require input file as command-line argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = "./output.md"
    model_id = os.getenv("MODEL_ID")
    data_dir = Path("output")
    data_dir.mkdir(exist_ok=True)

    # Read input file
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into sections
    sections = split_markdown_sections(text)
    input_file_stem = Path(input_path).stem

    # Store tuples of (date, processed_content)
    processed_entries = []
    max_workers = int(os.getenv("MAX_WORKERS", "4"))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_section_with_io, section, model_id, input_file_stem, data_dir)
            for section in sections
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing sections"):
            result = f.result()
            if result:
                processed_entries.append(result)

    # Sort all processed entries by date (alphabetically)
    processed_entries.sort(key=lambda x: x[0], reverse=True)
    curated_log = "\n\n".join([entry for _, entry in processed_entries])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(curated_log)
    print(f"Curated health log written to {output_path}")

if __name__ == "__main__":
    main()
