import os
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

# Load environment variables
load_dotenv()
api_key = os.getenv('key')
model = "deepseek-r1-distill-llama-70b"

# Initialize AI model
deepseek = ChatGroq(api_key=api_key, model_name=model)
parser = StrOutputParser()
deepseek_chain = deepseek | parser

# Load and process data from text file
loader = TextLoader('data.txt', encoding='utf-8')
data = loader.load()

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speaking speed

def speak(text):
    """Convert text to speech and output it."""
    engine.say(text)
    engine.runAndWait()

def listen():
    """Capture voice input and return recognized text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for your question...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            query = recognizer.recognize_google(audio)
            print(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError:
            print("Could not request results, check your internet connection.")
            return None

def chatbot():
    """Listen to user's question, get AI response, and speak it out."""
    question = listen()
    if question:
        template = f"""
        You are an AI-powered chatbot designed to provide 
        information and assistance based on the provided context.    
        Do not make things up.   
        Context: {data}
        Question: {question}
        """
        response = deepseek_chain.invoke(template)
        print(f"AI: {response}")
        speak(response)

if __name__ == "__main__":
    chatbot()
