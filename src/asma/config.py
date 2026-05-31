from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_INSTRUCTIONS = """You are an expert data extraction assistant. 
Your task is to extract key information from research articles to help create a structured dataset.

RULES:
1. Extract only the information explicitly stated in the text.
2. If a specific field is not mentioned anywhere in the text, output "Not specified". Do not guess or hallucinate data.
3. Keep the extracted values strictly concise. 
5. Output your response as simple key-value pairs in markdown formatted text [* key: value]."""

DEFAULT_FIELDS = [
    "Primary targeted bacteria species:",
    "Primary Bacterial Strain/isolate:",
    "Phage: [Include full name if available]",
    "Place of Sample collection:",
    "Phage isolation Sample:",
    "Phage Plaque characteristics:",
    "Phage TEM morphology:",
    "Phage TEM dimensions:",
    "Phage Taxonomy:",
    "Phage type (Lytic/Lysogenic/Engineered):",
    "All multiplicity of infection (MOI): [Scan the entire text, figures, and tables. List every numerical MOI value tested or mentioned, separated by commas]",
    "Optimal multiplicity of infection (MOI): [Analyze the reported MOI experiments. Identify and output the single MOI value that resulted in either: (1) the highest phage progeny/titer yield, or (2) the maximum reduction/lysis of target bacteria. If multiple experiments are shown, prioritize the standard MOI determination assay or time-kill assay results]",
    "Latent period (min):",
    "Burst size (phage/infected bacterium):",
    "Optimal Temperature (°C):",
    "Optimal pH:",
    "Phage Genome size (bp):",
    "Phage GC content (%):",
    "Phage Genome Accession/Bioproject:",
]

DEFAULT_EXTRACTION_PROMPT_LAYOUT = """{instructions}

#### Fields to be extracted:
{fields}

<document>
{document}
</document>"""

# Retained for legacy backward compatibility
DEFAULT_EXTRACTION_PROMPT_TEMPLATE = DEFAULT_EXTRACTION_PROMPT_LAYOUT.format(
    instructions=DEFAULT_INSTRUCTIONS,
    fields="\n".join(f"* {f.strip()}" for f in DEFAULT_FIELDS),
    document="{document}"
)

@dataclass
class AsmaConfig:
    """
    Configuration options for the asma library.
    """
    model_name: str = "google/gemma-4-e2b"
    ncbi_email: str = "asma@example.com"
    ncbi_tool: str = "asma_extractor"
    instructions: str = DEFAULT_INSTRUCTIONS
    fields: List[str] = field(default_factory=lambda: list(DEFAULT_FIELDS))
    prompt_layout: str = DEFAULT_EXTRACTION_PROMPT_LAYOUT

    # Legacy attribute fallback
    extraction_prompt_template: str = DEFAULT_EXTRACTION_PROMPT_TEMPLATE

    def build_prompt(self, document: str, fields: Optional[List[str]] = None) -> str:
        """
        Dynamically constructs the LLM extraction prompt by formatting the instructions,
        fields, and document content.
        
        Args:
            document: The document markdown string to parse.
            fields: Optional list of fields to extract. If not provided, defaults to configured fields.
            
        Returns:
            The fully constructed prompt string.
        """
        active_fields = fields if fields is not None else self.fields
        formatted_fields = "\n".join(f"* {f.strip()}" for f in active_fields)
        return self.prompt_layout.format(
            instructions=self.instructions,
            fields=formatted_fields,
            document=document
        )
