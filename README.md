# FAQ Chatbot

This is an FAQ chatbot built using FastAPI and a zero-shot classification model from Hugging Face Transformers. It categorizes user questions and provides relevant answers based on predefined FAQ data.

## Features
- Uses `typeform/distilbert-base-uncased-mnli` for zero-shot intent classification
- FastAPI-based backend with a `/chat/` endpoint
- CORS middleware enabled for cross-origin requests
- Answers FAQs from a structured JSON file

## Installation
### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/Steven-Leow/FAQ_ChatBot.git
   cd FAQ_ChatBot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have your FAQ data in `data/faq_data.json`.

## Usage
### Run the chatbot API
```bash
uvicorn main:app --host localhost --port 8000
```

### API Endpoint
**POST** `/chat/`
- Request Body:
  ```json
  {
    "message": "How do I reset my password?"
  }
  ```
- Response:
  ```json
  {
    "response": "Go to the settings page and click 'Reset Password'."
  }
  ```

## Future Improvements
- Improve accuracy with fine-tuned models
- Add support for multiple languages
- Implement a frontend UI

