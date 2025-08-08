# !/usr/bin/env python3
"""
Objective: Whenever someone uploads a doc or bunch of docs, we need to figure out what to do with them and suggest those actions to the user.


Idea:
1. Extract text from the document(s) using OCR or text extraction libraries.
2. Summarize the content using NLP techniques to get a gist of the document or use Classification model.
    2.1 I need to define features/actions as well to undertsand when to use which model.
    2.2 What should be the default or catch-all feature?
3. Based on the summary or classification, suggest actions like:
    A. 
    B.
    C. 
4. Present these suggestions to the user in a user-friendly interface.
"""

#Libraries
import os
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
LANGUAGE_MODEL = "gpt-4.1-2025-04-14"


class Document(BaseModel):
    """
    Pydantic model for summary.
    """
    summary: str = Field(
        ...,
        description="A brief summary of the document, written in 1-2 sentences. This should capture the main topic, parties, and purpose if possible."
    )

class DocumentProcessor:
    def __init__(self, input_path):
        self.input_path = input_path

    def extract_text(self):
        # Implement text extraction logic here
        # For Multiple files

        #For a txt file
        
        pass

    def summarize_content(text: str) -> Document:
        # Implement summarization logic using OpenAI or other NLP techniques
        system_prompt = """ You are a distinguised legal assistant. You will be provided with a chunk of 
        text from a legal document. Your task is to summarize the content in 1-2 sentences, 
        capturing the main topic and the purpose of the document. 
        """

        messages = [ 
            {
                "role": "system", 
                "content": system_prompt
                },
                     {
                         "role": "user", 
                         "content": text
                    } 
                     ]
        
        # Call the OpenAI API to get the summary
        response = client.responses.parse(
            model = LANGUAGE_MODEL,
            input = messages,
            text_format = Document
        )

        return response.output_parsed

    def classify_content(self, summary):
        # Implement classification logic to determine actions
        pass

    def suggest_actions(self, classification):
        # Suggest actions based on classification

        
        pass

def main():
    input_path = "data/2atlc-exhibit101c033117.txt"
    text = open(input_path, 'r', encoding='utf-8').read()
    result = DocumentProcessor.summarize_content(text = text)
    print("Summary:" , result.summary)


if __name__ == "__main__":
    main()
