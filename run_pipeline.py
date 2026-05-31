import os
import re
import sys
import json
import logging
from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm

from asma import (
    AsmaConfig,
    CrossrefResolver,
    PmcFetcher,
    LMStudioProvider,
    parse_bioc_to_llm_markdown,
    Evaluator,
    extract_doi_from_pdf
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("asma_runner")

def run_pipeline(pdf_name: str, config: AsmaConfig):
    # Ensure output directories exist
    os.makedirs("./json", exist_ok=True)
    os.makedirs("./md", exist_ok=True)
    
    pdf_path = Path(f"./pdf/{pdf_name}.pdf")
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)

    # 1. Extract DOI
    try:
        doi = extract_doi_from_pdf(pdf_path)
    except Exception as e:
        logger.error(f"DOI Extraction failed: {e}")
        sys.exit(1)

    # 2. Validate DOI
    resolver = CrossrefResolver()
    doi_meta = resolver.resolve_doi(doi)
    if doi_meta is None:
        logger.warning("DOI validation via Crossref failed or returned no metadata. Proceeding anyway...")
    else:
        logger.info("DOI validation via Crossref succeeded.")

    # 3. Fetch BioC JSON
    fetcher = PmcFetcher(email=config.ncbi_email, tool=config.ncbi_tool)
    try:
        bioc_data = fetcher.fetch_by_doi(doi)
        # Save fetched BioC JSON to file
        json_path = f"./json/{pdf_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(bioc_data, f, indent=4)
        logger.info(f"Successfully saved BioC JSON to {json_path}")
    except Exception as e:
        logger.error(f"Failed to fetch BioC JSON: {e}")
        sys.exit(1)

    # 4. Parse to Markdown
    try:
        markdown_content = parse_bioc_to_llm_markdown(bioc_data)
        md_path = f"./md/{pdf_name}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Successfully parsed and saved Markdown to {md_path}")
    except Exception as e:
        logger.error(f"Parsing BioC JSON to Markdown failed: {e}")
        sys.exit(1)

    # 5. Connect to Local LLM and Extract structured variables
    try:
        llm = LMStudioProvider(model_name=config.model_name)
        
        # Build prompt
        prompt = config.extraction_prompt_template.format(document=markdown_content)
        
        # Execute inference with a progress bar callback
        pbar = tqdm(total=100, bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}, {rate_fmt}]")
        def progress_callback(progress):
            current_val = progress * 100
            pbar.update(current_val - pbar.n)
            
        logger.info("Running structured extraction with local LLM...")
        raw_extraction = llm.respond(prompt, progress_callback=progress_callback)
        pbar.close()
        
        # Clean reasoning thought blocks if any
        from asma import clean_llm_response
        extracted_data = clean_llm_response(raw_extraction)
        
        result_path = f"./md/{pdf_name}_result.md"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(extracted_data)
        logger.info(f"Saved prediction output to {result_path}")
        
        print("\n=== Extraction Result ===")
        print(extracted_data)
        print("==========================\n")
        
    except Exception as e:
        logger.error(f"Inference/Extraction failed: {e}")
        sys.exit(1)

    # 6. Evaluation (conditional)
    ground_truth_path = Path(f"./md/{pdf_name}_gt.md")
    if ground_truth_path.exists():
        logger.info(f"Ground truth file found at {ground_truth_path}. Running quality evaluation...")
        try:
            with open(ground_truth_path, "r", encoding="utf-8") as f:
                ground_truth = f.read()
                
            evaluator = Evaluator(llm)
            comparison_report = evaluator.evaluate(extracted_data, ground_truth)
            
            comp_path = f"./md/{pdf_name}_comparison.md"
            with open(comp_path, "w", encoding="utf-8") as f:
                f.write(comparison_report)
                
            with open("comparison.md", "w", encoding="utf-8") as f:
                f.write(comparison_report)
                
            logger.info(f"Successfully saved comparison results to {comp_path} and global comparison.md")
            
            print("\n=== Evaluation Report ===")
            print(comparison_report)
            print("==========================\n")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
    else:
        logger.info(f"Ground truth file not found at {ground_truth_path}. Skipping evaluation.")

if __name__ == "__main__":
    # You can change the default PDF name to process
    pdf_name_arg = "36374021"
    
    if len(sys.argv) > 1:
        pdf_name_arg = sys.argv[1]
        
    config = AsmaConfig(
        model_name="google/gemma-4-e2b",
        ncbi_email="asma@example.com",
        ncbi_tool="asma_extractor"
    )
    
    run_pipeline(pdf_name_arg, config)
