from dotenv import load_dotenv; load_dotenv(override=True)
import os, re, sys
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil.parser import parse as date_parse
import argparse
import hashlib

def load_prompt(prompt_name):
    """Load a prompt from the prompts directory."""
    prompts_dir = Path(__file__).parent / "prompts"
    prompt_path = prompts_dir / f"{prompt_name}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file {prompt_path} does not exist.")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

PROCESS_SYSTEM_PROMPT = load_prompt("process.system_prompt")
VALIDATE_SYSTEM_PROMPT = load_prompt("validate.system_prompt")
VALIDATE_USER_PROMPT = load_prompt("validate.user_prompt")
SUMMARY_SYSTEM_PROMPT = load_prompt("summary.system_prompt")

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

def process(input_path):
    model_id = os.getenv("MODEL_ID")

    # Create output path
    data_dir = Path("output") / input_path.stem
    data_dir.mkdir(parents=True, exist_ok=True)

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
                    {
                        "role": "system",
                        "content": PROCESS_SYSTEM_PROMPT
                    },
                    {
                        "role": "user", 
                        "content": raw_section
                    }
                ],
                max_tokens=2048,
                temperature=0.0
            )
            processed_section = completion.choices[0].message.content.strip()

            # Run LLM to validate the processed section
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system", 
                        "content": VALIDATE_SYSTEM_PROMPT
                    },
                    {
                        "role": "user", 
                        "content": VALIDATE_USER_PROMPT.format(
                            raw_section=raw_section,
                            processed_section=processed_section
                        )
                    }
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
        if count != 1:
            print(f"Section does not contain exactly one '###':\n{section}")
            sys.exit(1)

    # Assert no duplicate dates in sections
    dates = [extract_date_from_section(section) for section in sections]
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
    with open(data_dir / "output.md", "w", encoding="utf-8") as f: f.write(processed_text)
    print(f"Saved processed health log to {data_dir / 'output.md'}")

    # Write the summary using the LLM
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system", 
                "content": SUMMARY_SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": processed_text
            }
        ],
        max_tokens=2048,
        temperature=0.0
    )
    summary = completion.choices[0].message.content.strip()
    with open(data_dir / "summary.md", "w", encoding="utf-8") as f: f.write(summary)
    print(f"Saved processed health summary to {data_dir / 'summary.md'}")

def main():
    parser = argparse.ArgumentParser(description="Health log parser and validator")
    parser.add_argument("input_file", help="Original input file")
    
    args = parser.parse_args()
    process(args.input_file)

if __name__ == "__main__":
    main()
