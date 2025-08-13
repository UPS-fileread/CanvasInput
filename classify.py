#!/usr/bin/env python3
"""
Classify and summarize legal documents using OpenAI LLM and Pydantic for schema validation.
"""
import argparse
import sys
import warnings
import os
from enum import Enum
import json
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator
from langfuse import get_client
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
langfuse_client = get_client()

# Load classification definitions from JSON config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "definitions.json")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

# definitions_map is a dict mapping stringified IDs to names
definitions_map: dict[int, str] = {
    int(object_id): name
    for object_id, name in cfg.items()
}

# Dynamically generate Category enum from definitions_map (top-level categories have IDs divisible by 10000000)
Category = Enum(
    "Category",
    [(name.replace(" ", "_"), name) for id, name in definitions_map.items() if id % 100000000000000000000000 == 0]
)

# Build subcategories map: group definitions_map entries by category id
subcategories_map: dict[str, list[str]] = {}
for id, name in definitions_map.items():
    # Skip top-level category IDs
    if id % 100000000000000000000000 == 0:
        continue
    # Compute the parent category ID by stripping lower-order digits
    cat_id = (id // 100000000000000000000000) * 100000000000000000000000
    parent = definitions_map.get(cat_id)
    if parent:
        subcategories_map.setdefault(parent, []).append(name)

# Dynamically generate Subcategory enum from subcategories_map
Subcategory = Enum(
    "Subcategory",
    [(sc.replace(" ", "_"), sc) for subs in subcategories_map.values() for sc in subs]
)

# Model and categories
LANGUAGE_MODEL = 'gpt-4.1-2025-04-14'

class ClassificationResult(BaseModel):
    """
    Pydantic model for classification results.
    """
    category: Category = Field(
        ...,
        description="One of the predefined legal-context categories",
        example="Regulatory"
    )
    subcategory: Subcategory = Field(
        ...,
        description="One of the predefined subcategories for the chosen category. Use the provided subcategory definitions to ensure the best fit.",
        example="Inspection Report"
    )
    summary: str = Field(
        ...,
        description="A brief summary of the document, written in 1-2 sentences. This should capture the main topic, parties, and purpose if possible."
    )
    key_themes: list[str] = Field(
        ...,
        description="A list of 3 concise bullet points highlighting the most important facts, events, obligations, or issues in the document. These should serve as an extended, detailed highlight of the document—covering key events, important parties, main legal or factual points, and anything a litigator would want to quickly grasp."
    )
    @model_validator(mode="after")
    def check_subcategory_matches_category(cls, model):
        category_key = model.category.value if hasattr(model.category, "value") else model.category
        allowed = subcategories_map.get(category_key, [])
        if model.subcategory not in allowed:
            raise ValueError(
                f"Subcategory '{model.subcategory}' is not valid for category '{category_key}'"
            )
        return model
    class Config:
        use_enum_values = True
        validate_assignment = True

def classify_context(text: str) -> ClassificationResult:
    """
    Classify and summarize the given document text using an LLM.
    Returns a ClassificationResult object.
    """
    # Fetch the system prompt template from Langfuse by prompt name
    prompt_template = langfuse_client.get_prompt("classification/main").get_langchain_prompt()
    # Build messages for LLM: system prompt from Langfuse and user text
    messages = [
        {
            "role": "system", 
            "content": prompt_template
        },
        {
            "role": "user",
            "content": text
        },
    ]

    # Make the API call to classify the text
    resp = client.beta.chat.completions.parse(
        model=LANGUAGE_MODEL,
        messages=messages,
        response_format=ClassificationResult
    )

    parsed = resp.choices[0].message.parsed
    category_key = str(parsed.category)
    subcategory_value = str(parsed.subcategory)

    # First validation
    if subcategory_value not in subcategories_map.get(category_key, []):
        # Retry once
        resp = client.beta.chat.completions.parse(
            model=LANGUAGE_MODEL,
            messages=messages,
            response_format=ClassificationResult
        )
        parsed = resp.choices[0].message.parsed
        category_key = str(parsed.category)
        subcategory_value = str(parsed.subcategory)

        # Second validation
        if subcategory_value not in subcategories_map.get(category_key, []):
            # Assign "other" for both category and subcategory
            parsed.category = Category.other
            parsed.subcategory = Subcategory.Other

    return parsed

def main():
    """
    Command-line interface for classifying a TXT file into a legal context and subcategory.
    """
    parser = argparse.ArgumentParser(
        description="Classify a TXT file into a legal context and subcategory."
    )
    parser.add_argument('txt_file', help="Path to the TXT file to classify")
    args = parser.parse_args()

    # Read the input text file
    try:
        text = open(args.txt_file, 'r', encoding='utf-8').read()
    except Exception as e:
        print(f"Error reading '{args.txt_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # Classify and print results
    try:
        result = classify_context(text)
        print(f"Category   : {result.category}")
        print(f"Subcategory: {result.subcategory}")
        print(f"Summary    : {result.summary}")
        print("Key Themes :")
        for idx, theme in enumerate(result.key_themes, 1):
            print(f"  {idx}. {theme}")
    except ValidationError as e:
        warnings.warn(f"Failed to validate LLM response: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()