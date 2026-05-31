# %% [markdown]
# # Unified PDF-to-LLM Structured Extraction Pipeline
# 
# This notebook integrates three steps into a single, automated workflow:
# 1. **Fetch BioC JSON**: Extract DOI from the PDF, validate it, resolve PMC/PubMed IDs, and retrieve BioC JSON from PMC-NIH API.
# 2. **Parse to Markdown**: Convert the BioC JSON to clean, LLM-optimized Markdown (including parsing XML tables to CSV style).
# 3. **Local LLM Inference**: Run information extraction using LM Studio with `google/gemma-4-e2b` to extract structured research variables.
# 4. **Conditional Comparison**: If a ground truth Markdown file is available, run a local LLM assessment comparing the prediction to the truth and generate a report.

# %%
# # Step 1: Install Dependencies
# %pip install crossrefapi backoff PyMuPDF requests bioc bs4 lmstudio tqdm lxml

# %%
# Step 2: Imports
import os
import re
import csv
import io
import json
import requests
from pathlib import Path
import fitz  # PyMuPDF
from crossref.restful import Works
from bioc.biocjson.decoder import parse_collection
from bs4 import BeautifulSoup
import lmstudio as lms
from IPython.display import Markdown
from tqdm import tqdm

# %%
# Step 3: Configure Input PDF File Name
pdf_name = "36374021"

# %%
# Step 4: Text Cleaning and Parsing Helpers
def clean_text(text):
    # Remove bracketed citations like [1], [1, 2], [3-5]
    cleaned = re.sub(r'\[\d+(?:[\s,\u2013-]+\d+)*\]', '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

def extract_metadata_llm(infons):
    lines = []
    
    doi = infons.get("article-id_doi")
    if doi:
        lines.append(f"DOI: https://doi.org/{doi}")
        
    pmc = infons.get("article-id_pmc")
    pmid = infons.get("article-id_pmid")
    if pmc: lines.append(f"PMC: {pmc}")
    if pmid: lines.append(f"PMID: {pmid}")
    
    volume = infons.get("volume")
    issue = infons.get("issue")
    year = infons.get("year")
    journal_parts = []
    if volume: journal_parts.append(f"Volume {volume}")
    if issue: journal_parts.append(f"Issue {issue}")
    if year: journal_parts.append(f"({year})")
    if journal_parts:
        lines.append(f"Published: {', '.join(journal_parts)}")
        
    authors = []
    for i in range(50):
        val = infons.get(f"name_{i}")
        if not val:
            break
        parts = dict(item.split(":", 1) for item in val.split(";") if ":" in item)
        name = f"{parts.get('given-names', '')} {parts.get('surname', '')}".strip()
        if name and name not in authors:
            authors.append(name)
    if authors:
        lines.append(f"Authors: {', '.join(authors)}")

    kwd = infons.get("kwd")
    if kwd:
        lines.append(f"Keywords: {kwd}")
        
    return lines

def parse_xml_table_to_csv(xml_str):
    soup = BeautifulSoup(xml_str, "lxml-xml")
    table = soup.find("table")
    if not table:
        soup = BeautifulSoup(f"<table>{xml_str}</table>", "lxml-xml")
        table = soup.find("table")
    if not table:
        return ""

    inline_tags = {"sup", "sub", "italic", "bold", "xref"}
    
    def cell_text(cell):
        parts = []
        for child in cell.descendants:
            if isinstance(child, str):
                parts.append(child)
            elif child.name not in inline_tags:
                parts.append(" ")
        raw = "".join(parts)
        return re.sub(r"\s{2,}", " ", raw).strip()

    rows = []
    thead = table.find("thead")
    if thead:
        header_row = thead.find("tr")
        if header_row:
            rows.append([cell_text(c) for c in header_row.find_all(["th", "td"])])

    if not rows:
        first_tr = table.find("tr")
        if first_tr:
            rows.append([cell_text(c) for c in first_tr.find_all(["th", "td"])])

    if not rows:
        return ""

    tbody = table.find("tbody")
    body_rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]
    
    for tr in body_rows:
        cells = [cell_text(c) for c in tr.find_all(["td", "th"])]
        rows.append(cells)

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().strip()

# %%
# Step 5: Phase 1 - Extract DOI & Fetch BioC JSON
os.makedirs("./json", exist_ok=True)
os.makedirs("./md", exist_ok=True)

pdf_file = f"./pdf/{pdf_name}.pdf"
print(f"Opening PDF file: {pdf_file}")

with fitz.open(pdf_file) as pdf:
    text = ""
    for i in range(min(2, len(pdf))):
        text += pdf[i].get_text() or ""
    
    doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Z0-9]+'
    match = re.search(doi_pattern, text, re.IGNORECASE)

if match:
    doi = match.group(0)
    print(f"Found DOI: {doi}")
else:
    raise ValueError("No DOI found in the first two pages of the PDF.")

# Validate DOI via Crossref
print("Validating DOI on Crossref...")
works = Works()
result = works.doi(doi)

if result is None:
    print("Warning: DOI validation failed. Attempting to retrieve IDs anyway...")
else:
    print("DOI validation succeeded.")

# Resolve DOI to PMCID/PMID
import time
id_conv_url = f"https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids={doi}&format=json&tool=amas_extractor&email=amas@example.com"
for attempt in range(3):
    try:
        resp = requests.get(id_conv_url)
        id_data = resp.json()
        break
    except Exception as e:
        if attempt == 2: raise e
        print(f"PMC API attempt {attempt+1} failed, retrying in 2 seconds...")
        time.sleep(2)
record = id_data.get('records', [{}])[0]
pmcid = record.get('pmcid')
pmid = record.get('pmid')

if pmcid:
    target_id = pmcid
    api_endpoint = "pmcoa.cgi"
    print(f"Resolved target ID to PMCID: {target_id}")
elif pmid:
    target_id = pmid
    api_endpoint = "pubmed.cgi"
    print(f"Resolved target ID to PMID: {target_id}")
else:
    raise ValueError("DOI conversion failed to yield a PMCID or PMID.")

# Fetch BioC JSON
bioc_url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/{api_endpoint}/BioC_json/{target_id}/unicode"
print(f"Fetching BioC JSON from {bioc_url}...")
response = requests.get(bioc_url)

if response.status_code == 200:
    bioc_data = response.json()
    json_name = f"./json/{pdf_name}.json"
    with open(json_name, "w") as f:
        json.dump(bioc_data, f, indent=4)
    print(f"Successfully saved BioC JSON to {json_name}")
else:
    raise ValueError(f"BioC JSON retrieval failed with status code: {response.status_code}")

# %%
# Step 6: Phase 2 - Parse BioC JSON to LLM-Optimized Markdown
json_path = f"./json/{pdf_name}.json"
print(f"Loading BioC JSON from {json_path}...")
with open(json_path, "r", encoding="utf-8") as fh:
    raw = json.load(fh)

obj = raw[0] if isinstance(raw, list) else raw
collection = parse_collection(obj)
doc = collection.documents[0]
print(f"Loaded document: {doc.id}")

output_lines = []

for p in doc.passages:
    pt = p.infons.get("type", "")
    st = p.infons.get("section_type", "")
    text = p.text.strip()
    
    if not text:
        continue
        
    # Front Matter & Metadata
    if pt == "front":
        output_lines.append(f"# {text}\n")
        meta = extract_metadata_llm(p.infons)
        if meta:
            output_lines.extend(meta)
            output_lines.append("")
        continue
        
    # Skip Reference Section
    if st == "REF" or (pt in ("title", "title_1") and st == "REF") or pt == "ref":
        continue
        
    # Headings
    if pt in ("abstract_title_1", "title_1"):
        output_lines.append(f"## {text}\n")
        continue
    if pt == "title_2":
        output_lines.append(f"### {text}\n")
        continue
    if pt == "title_3":
        output_lines.append(f"#### {text}\n")
        continue
        
    # Figure & Table Captions
    if pt == "fig_caption":
        fig_id = p.infons.get("id", "")
        fig_num = fig_id.replace("fig", "") if fig_id.startswith("fig") else fig_id
        label = f"Figure {fig_num}" if fig_num else "Figure"
        output_lines.append(f"{label}: {clean_text(text)}\n")
        continue
    if pt == "table_caption":
        tab_id = p.infons.get("id", "")
        tab_num = tab_id.replace("tab", "") if tab_id.startswith("tab") else tab_id
        label = f"Table {tab_num}" if tab_num else "Table"
        output_lines.append(f"{label}: {clean_text(text)}\n")
        continue
        
    # Tables (XML to CSV conversion)
    if pt == "table":
        xml_str = p.infons.get("xml", "")
        if xml_str:
            csv_data = parse_xml_table_to_csv(xml_str)
            if csv_data:
                output_lines.append(csv_data + "\n")
        elif text:
            output_lines.append(clean_text(text) + "\n")
        continue
        
    # Fallback to general paragraph parsing
    output_lines.append(clean_text(text) + "\n")

output_content = "\n".join(output_lines)
output_path = f"./md/{pdf_name}.md"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(output_content)

print(f"Successfully parsed and saved Markdown to {output_path}")

# %%
# Step 7: Phase 3 - Structured Data Extraction with Local LLM (LM Studio)
print("Connecting to local LM Studio...")
model = lms.llm("google/gemma-4-e2b")

input_md_path = f"./md/{pdf_name}.md"
with open(input_md_path, "r", encoding="utf-8") as f:
    paper = f.read()

# Structured extraction prompt (v2)
sys_inst = f"""You are an expert data extraction assistant. 
Your task is to extract key information from research articles to help create a structured dataset.

RULES:
1. Extract only the information explicitly stated in the text.
2. If a specific field is not mentioned anywhere in the text, output "Not specified". Do not guess or hallucinate data.
3. Keep the extracted values strictly concise. 
5. Output your response as simple key-value pairs in markdown formatted text [* key: value].

#### Fields to be extracted:
* Primary targeted bacteria species:
* Primary Bacterial Strain/isolate:
* Phage: [Include full name if available]
* Place of Sample collection:
* Phage isolation Sample:
* Phage Plaque characteristics:
* Phage TEM morphology:
* Phage TEM dimensions:
* Phage Taxonomy:
* Phage type (Lytic/Lysogenic/Engineered):
* All multiplicity of infection (MOI): [Scan the entire text, figures, and tables. List every numerical MOI value tested or mentioned, separated by commas]
* Optimal multiplicity of infection (MOI): [Analyze the reported MOI experiments. Identify and output the single MOI value that resulted in either: (1) the highest phage progeny/titer yield, or (2) the maximum reduction/lysis of target bacteria. If multiple experiments are shown, prioritize the standard MOI determination assay or time-kill assay results]
* Latent period (min):
* Burst size (phage/infected bacterium):
* Optimal Temperature (°C):
* Optimal pH:
* Phage Genome size (bp):
* Phage GC content (%):
* Phage Genome Accession/Bioproject:

<document>\n{paper}\n</document>
"""

chat = lms.Chat()
chat.add_user_message(sys_inst)

prompt_template_str = model.apply_prompt_template(chat)
token_count = len(model.tokenize(prompt_template_str))
print(f"Token count: {token_count}")
print(f"Context length: {model.get_context_length()}")

# Executing Inference with live progress
pbar = tqdm(total=100, bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}, {rate_fmt}]")
def progress_callback(progress):
    current_val = progress * 100
    pbar.update(current_val - pbar.n)

result = model.respond(
    chat,
    on_prompt_processing_progress=progress_callback
)
pbar.close()

message = result.content

# Extract and display the reasoning block (thought) and extraction response
if "<|channel>thought" in message and "<channel|>" in message:
    thought_part, answer_part = message.split("<channel|>", 1)
    thought = thought_part.replace("<|channel>thought", "").strip()
    answer = answer_part.strip()
    # display(Markdown(f"### Thought\n{thought}\n"))
    # display(Markdown(f"### Response\n{answer}"))
else:
    answer = message.strip()
    display(Markdown(f"### Response\n{message}"))

# Save extracted info to result file
result_path = f"./md/{pdf_name}_result.md"
with open(result_path, "w", encoding="utf-8") as f:
    f.write(answer)
print(f"Saved prediction output to {result_path}")

# %%
# Step 8: Phase 4 - Conditional Ground Truth Quality Evaluation
ground_truth_path = f"./md/{pdf_name}_gt.md"

if os.path.exists(ground_truth_path):
    print(f"Ground truth file found at {ground_truth_path}. Running comparison evaluation...")
    
    # Load prediction and ground truth content
    with open(f"./md/{pdf_name}_result.md", "r", encoding="utf-8") as f:
        prediction = f.read()
        
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = f.read()
        
    comparison_prompt = f"""
You are an expert LLM quality assessor. 
Your task is to compare a predicted result against its ground truth and assign a holistic quality score.

**Scoring Scale:**
*   **0:** Complete hallucination or completely incorrect.
*   **10:** Picture-perfect accuracy.

**Input Data:**
---
### Predicted Result
<document>
{prediction}
</document>

### Ground Truth
<document>
{ground_truth}
</document>
---

**Instructions for Output:**
Analyze each key-value of the Predicted Result against the ground truth and generate a single result formatted strictly as a Markdown table. 
Do not include any introductory or concluding sentences outside of the table.

**Required Output Format (MUST follow this structure):**
* Overall score: [0 - 10]

| Key | Prediction | Truth | Score [0 - 10] | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| [Key] | [prediction] | [ground_truth] | [Calculated Score] | [Your detailed analysis of the discrepancy] |
"""
    
    print("Comparing prediction against ground truth via LLM...")
    comparison_result = model.respond(comparison_prompt)
    comp_message = comparison_result.content
    
    # Remove thought tag block from comparison output if present
    if "<|channel>thought" in comp_message and "<channel|>" in comp_message:
        _, comp_message = comp_message.split("<channel|>", 1)
        comp_message = comp_message.strip()
        
    display(Markdown(f"### Comparison Result\n{comp_message}"))
    
    # Write comparison files
    comp_path = f"./md/{pdf_name}_comparison.md"
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write(comp_message)
    
    with open("comparison.md", "w", encoding="utf-8") as f:
        f.write(comp_message)
        
    print(f"Successfully saved comparison results to {comp_path} and global comparison.md")
else:
    print(f"Ground truth file not found at {ground_truth_path}. Exiting gracefully without evaluation.")


