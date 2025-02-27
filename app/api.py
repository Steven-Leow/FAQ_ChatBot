from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import Chatbot  # Import the Chatbot class
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the Chatbot instance
chatbot = Chatbot(faq_data_path='data/faq_data.json')

class UserMessage(BaseModel):
    message: str

@app.post("/chat/")
async def chat_with_bot(user_message: UserMessage):
    # Get the category and topic based on user message
    category, topic = chatbot.classify_intent(user_message.message)

    if category is None or topic is None:
        # If no valid category or topic, return a default response
        return {"response": "Sorry, I couldn't understand your question. Please try rephrasing."}
    
    # Fetch the answer from the chatbot
    answer = chatbot.get_answer(category, topic)
    return {"response": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
