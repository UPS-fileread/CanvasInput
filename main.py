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

from openai import OpenAI


def main():
    input_path = "data/"




if __name__ == "__main__":
    main()
