# 🩺 health-log-parser

<p align="center">
  <img src="logo.png" alt="Logo" width="400"/>
</p>

🔹 **AI-powered tool for structuring and auditing personal health logs**  

## 📖 Overview

health-log-parser transforms unstructured or semi-structured health journal entries into clean, standardized Markdown summaries. It uses an LLM (via OpenRouter) to extract and format details about medical visits, medications, and symptoms. The tool also audits the structured output, reporting any clinical data missing from the conversion.

This project is ideal for patients, caregivers, or clinicians who want to organize health notes and ensure no important information is lost in the process.

## 🚀 Installation

```bash
pipx install . --force
```

- Requires Python 3.8+
- Set up your `.env` file (see `.env.example`) with your OpenRouter API key and preferred model

## 🛠️ Usage

**Basic usage:**

```bash
python main.py <your_health_log.md>
```

- Input: A Markdown file with health log entries (can be unstructured)
- Output: 
  - `output/output.md` — curated, structured health log
  - `output/clinical_data_missing_report.md` — report of any clinical data missing from the structured output

**Environment variables (`.env`):**

- `MODEL_ID` — LLM model to use (e.g., `openai/gpt-4.1`)
- `OPENROUTER_API_KEY` — your OpenRouter API key
- `MAX_WORKERS` — number of parallel processing threads (default: 1)

**Example workflow:**

1. Prepare your `.env` file with API credentials.
2. Run the tool on your health log:
   ```bash
   python main.py my_health_journal.md
   ```
3. Review the structured log and missing data report in the `output/` directory.

## 📄 License

This project is licensed under the [MIT License](LICENSE).