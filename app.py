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
st.title("SkillBridge Question Answering Chatbot")

# Predefined questions for SkillBridge
st.subheader("Try asking one of these questions:")
predefined_questions = [
    "What courses does SkillBridge offer for content writing?",
    "How can I improve my graphic designing skills through SkillBridge?",
    "Are there any beginner-friendly courses available in SkillBridge?",
    "What is the duration of the content writing course?",
    "Does SkillBridge provide certifications upon course completion?",
    "What are the prerequisites for enrolling in a graphic design course?",
    "How can I access course materials after enrolling in SkillBridge?",
    "Are there any community forums for SkillBridge students?",
    "What payment options are available for SkillBridge courses?",
    "Can I get personalized feedback on my projects from instructors at SkillBridge?",
]

for question in predefined_questions:
    if st.button(question):
        st.session_state.user_question = question  # Store the question in the session state
        st.experimental_rerun()  # Rerun the app to process the question

# User input for question
user_question = st.text_input("Or ask your own question:", value=st.session_state.get('user_question', ''))

# Context for question answering (you can update this based on your use case)
context = """
SkillBridge offers a variety of courses in freelance skills, including content writing and graphic designing. 
These courses are designed for beginners and experienced individuals alike, with options for certifications 
upon completion. Students have access to course materials and can participate in community forums to enhance 
their learning experience.
"""

if user_question:
    # Get a response from the Groq LLM for a general conversational answer
    groq_response = get_groq_response(user_question)
    st.write(f"**Groq conversational response:** {groq_response}")

    # Extract a specific answer from the context using the Hugging Face model
    answer = ask_question(context, user_question)
    st.write(f"**Answer from deepset/roberta-base-squad2:** {answer}")
