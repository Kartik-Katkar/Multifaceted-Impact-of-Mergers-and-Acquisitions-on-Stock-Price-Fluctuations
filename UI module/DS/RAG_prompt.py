import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

def make_rag_prompt(query, relevant_passages):
    # Join all relevant passages into a single string
    combined_passages = " ".join(relevant_passages)
    escaped_passages = combined_passages.replace("'", "").replace('"', "").replace("\n", " ")

    prompt = ("""Do not give empty output. You are a helpful and informative bot that answers questions using text from the reference passage included below. \
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
If the passage is irrelevant to the answer, generate answer based on the which you were trained on Do not apologise give something.
QUESTION: '{query}'
PASSAGE: '{relevant_passages}'

ANSWER:
""").format(query=query, relevant_passages=escaped_passages)

    return prompt

def generate_answer(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text