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

Example output:

#### 2023-04-12

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

#### 2023-04-11

- Colonoscopy performed by **Dr. Jones (Gastroenterologist)** at **City Hospital**
    - Findings:
        - No polyps found, normal mucosa.
        - Biopsy taken for further analysis.
    - Notes:
        - Follow-up in 2 weeks for biopsy results.
        - Preparation was difficult, but manageable.
"""

# Split input text into markdown sections by date
def split_markdown_sections(text):

    return [s.strip() for s in re.split(r'(?=^###\s+\d{4}-\d{2}-\d{2})', text, flags=re.MULTILINE) if s.strip()]

# Extract date from section header, tolerant to different formats
def extract_date_from_section(section):
    header = section.strip().splitlines()[0].lstrip("#").strip()
    for token in re.split(r'\s+', header):
        try:
            return date_parse(token, fuzzy=False, dayfirst=False, yearfirst=True).strftime("%Y-%m-%d")
        except Exception:
            continue
    return "unknown"

# Run LLM to process a section, return formatted markdown
def process_section(section, model_id):
    lines = section.strip().splitlines()
    if not lines or not lines[0].startswith("###"): return None
    if not any(line.strip() for line in lines[1:]): return None
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

# Handle I/O for a section: write raw, check processed, or process if needed
def process_section_with_io(section, model_id, input_file_stem, data_dir):
    date = extract_date_from_section(section)
    raw_file = data_dir / f"{input_file_stem}_{date}.raw.md"
    processed_file = data_dir / f"{input_file_stem}_{date}.processed.md"
    write_raw = not (raw_file.exists() and raw_file.read_text(encoding="utf-8") == section)
    if write_raw:
        raw_file.write_text(section, encoding="utf-8")
    if not write_raw and processed_file.exists():
        processed_content = processed_file.read_text(encoding="utf-8").strip()
        if processed_content:
            return (date, processed_content)
    formatted = process_section(section, model_id)
    if formatted:
        processed_file.write_text(formatted, encoding="utf-8")
        return (date, formatted)
    return None

def get_file_short_hash(filepath):
    with open(filepath, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return file_hash[:8]

def extract_task(input_path, output_path, model_id, data_dir):
    # Read and split input file into sections
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    sections = split_markdown_sections(text)
    input_file_stem = Path(input_path).stem

    processed_entries, to_process = [], []
    for section in sections:
        date = extract_date_from_section(section)
        raw_file = data_dir / f"{input_file_stem}_{date}.raw.md"
        processed_file = data_dir / f"{input_file_stem}_{date}.processed.md"
        error_file = data_dir / f"{input_file_stem}_{date}.errors.md"
        raw_file.write_text(section, encoding="utf-8")
        raw_digest = hashlib.sha256(section.encode("utf-8")).hexdigest()
        # Check if processed file exists and its first line matches raw digest
        needs_processing = True
        if processed_file.exists():
            with processed_file.open(encoding="utf-8") as pf:
                lines = pf.readlines()
                if lines and lines[0].strip() == raw_digest:
                    needs_processing = False
        # If errors file exists and does not contain $OK$, force reprocessing
        if error_file.exists():
            with error_file.open(encoding="utf-8") as ef:
                lines = ef.readlines()
                if not any("$OK$" in line for line in lines):
                    needs_processing = True
        if needs_processing:
            to_process.append((section, raw_digest, date, raw_file, processed_file, error_file))
        else:
            # Read processed content (skip digest line)
            with processed_file.open(encoding="utf-8") as pf:
                lines = pf.readlines()
                processed_content = "".join(lines[1:]).strip()
                if processed_content:
                    processed_entries.append((date, processed_content))

    max_workers = int(os.getenv("MAX_WORKERS", "4"))
    def process_and_write(section, raw_digest, date, raw_file, processed_file, error_file):
        formatted = process_section(section, model_id)
        if formatted:
            # Write processed file with digest as first line
            processed_file.write_text(f"{raw_digest}\n{formatted.strip()}\n", encoding="utf-8")
            return (date, formatted.strip())
        return None

    if to_process:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_and_write, section, raw_digest, date, raw_file, processed_file, error_file)
                for (section, raw_digest, date, raw_file, processed_file, error_file) in to_process
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Processing sections"):
                result = f.result()
                if result:
                    processed_entries.append(result)

    # Sort and write final curated log
    processed_entries.sort(key=lambda x: x[0], reverse=True)
    curated_log = "\n\n".join([entry for _, entry in processed_entries])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(curated_log)
    print(f"Curated health log written to {output_path}")

def validate_extraction_task(data_dir, model_id):
    # For each .raw.md and .processed.md pair in data_dir, compare and write .errors.md if needed
    raw_files = list(data_dir.glob("*.raw.md"))

    def validate_one(raw_file):
        processed_file = data_dir / f"{raw_file.stem.replace('.raw', '')}.processed.md"
        if not processed_file.exists():
            return None
        input_text = raw_file.read_text(encoding="utf-8")
        output_text = processed_file.read_text(encoding="utf-8")
        
        # Compute hashes
        raw_hash = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
        processed_hash = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        error_file = data_dir / f"{raw_file.stem.replace('.raw', '')}.errors.md"
        
        # Check if error file exists and hashes match
        needs_validation = True
        if error_file.exists():
            first_line = error_file.open(encoding="utf-8").readline().strip()
            if first_line == f"{raw_hash};{processed_hash}":
                needs_validation = False
        if not needs_validation:
            return None
        system_prompt = (
            "You are a clinical data auditor. The user will provide two files: "
            "the first is the original health log (possibly unstructured), and the second is a curated/structured version. "
            "Your job is to identify and list any clinical data (symptoms, medications, medical visits, test results, dates, etc.) "
            "that is present in the original file but missing or omitted in the curated file. "
            "Be specific: for each missing item, quote the relevant text from the original and explain what is missing in the curated version. "
            "If nothing is missing, reply only with '$OK$' (without quotes) and do not add any other text."
        )
        user_prompt = (
            "Original health log (input section):\n-----\n"
            f"{input_text}\n-----\n"
            "Curated health log (output section):\n-----\n"
            f"{output_text}\n-----\n"
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
        error_content = completion.choices[0].message.content.strip()
        error_file.write_text(f"{raw_hash};{processed_hash}\n{error_content}", encoding="utf-8")
        return error_file

    max_workers = int(os.getenv("MAX_WORKERS", "4"))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(validate_one, raw_file) for raw_file in raw_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Validating extraction"):
            error_file = f.result()
            if error_file:
                print(f"Wrote error file: {error_file}")

def main():
    parser = argparse.ArgumentParser(description="Health log parser and validator")
    subparsers = parser.add_subparsers(dest="task", required=True)

    # Extract subcommand
    extract_parser = subparsers.add_parser("extract", help="Extract and curate health log")
    extract_parser.add_argument("input_file", help="Input file to process")

    # Validate extraction subcommand
    validate_parser = subparsers.add_parser("validate_extraction", help="Validate extraction by comparing input and output files")
    validate_parser.add_argument("input_file", help="Original input file")
    
    args = parser.parse_args()
    model_id = os.getenv("MODEL_ID")

    if args.task == "extract":
        input_path = args.input_file
        short_hash = get_file_short_hash(input_path)
        data_dir = Path("output") / short_hash
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / "output.md"
        extract_task(input_path, output_path, model_id, data_dir)
    elif args.task == "validate_extraction":
        input_path = args.input_file
        short_hash = get_file_short_hash(input_path)
        data_dir = Path("output") / short_hash
        assert data_dir.exists(), f"Data directory {data_dir} does not exist. Please run extraction first."
        validate_extraction_task(data_dir, model_id)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
