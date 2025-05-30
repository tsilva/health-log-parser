from dotenv import load_dotenv; load_dotenv(override=True)
import os, re, sys
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil.parser import parse as date_parse
import argparse
import hashlib

# Load prompts from external files
PROMPTS_DIR = Path(__file__).parent / "prompts"
with open(PROMPTS_DIR / "process.system_prompt.md", "r", encoding="utf-8") as f:
    process_system_prompt = f.read()
with open(PROMPTS_DIR / "validate.system_prompt.md", "r", encoding="utf-8") as f:
    validate_system_prompt = f.read()
with open(PROMPTS_DIR / "validate.user_prompt.md", "r", encoding="utf-8") as f:
    validate_user_prompt = f.read()

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

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
                    {"role": "system", "content": process_system_prompt},
                    {"role": "user", "content": raw_section}
                ],
                max_tokens=2048,
                temperature=0.0
            )
            processed_section = completion.choices[0].message.content.strip()

            # Run LLM to validate the processed section
            system_prompt = validate_system_prompt
            user_prompt = validate_user_prompt.format(
                raw_section=raw_section,
                processed_section=processed_section
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
