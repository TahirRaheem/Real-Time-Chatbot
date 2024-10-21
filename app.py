import streamlit as st
from groq import Groq
from transformers import pipeline

# Initialize Groq client with the API key directly (replace with your actual API key)
client = Groq(
    api_key="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",  # Replace this with your Groq API key
)

# Initialize the deepset/roberta-base-squad2 pipeline for question answering
question_answerer = pipeline("question-answering", model="deepset/roberta-base-squad2")

def get_groq_response(prompt):
    # Get a response from the Groq LLM
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",  # Example model; you can change as needed
    )
    return chat_completion.choices[0].message.content

def ask_question(context, question):
    # Use deepset/roberta-base-squad2 for answering the question based on the provided context
    result = question_answerer(question=question, context=context)
    return result['answer']

# Streamlit app
st.title("Question Answering Chatbot")

# User input for question
user_question = st.text_input("Ask your question:")

# Context for question answering (you can update this based on your use case)
context = """
Fast language models like GPT and LLAMA are crucial for modern NLP tasks. They allow real-time processing and 
generate human-like responses, which is vital for applications such as chatbots, virtual assistants, and 
content generation.
"""

if user_question:
    # Get a response from the Groq LLM for a general conversational answer
    groq_response = get_groq_response(user_question)
    st.write(f"**Groq conversational response:** {groq_response}")

    # Extract a specific answer from the context using the Hugging Face model
    answer = ask_question(context, user_question)
    st.write(f"**Answer from deepset/roberta-base-squad2:** {answer}")
