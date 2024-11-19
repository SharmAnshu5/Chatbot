# HR Queries Chatbot Backend

This repository contains the backend code for an HR Queries Chatbot. The chatbot is designed to handle employee HR-related queries such as leave policies, working hours, salary inquiries, and other HR information.

## Features
- Responds to employee queries about HR policies, procedures, and other commonly asked HR-related questions.
- Uses a simple rule-based or intent-based response generation system.
- Built with Flask to handle HTTP requests.
  
## Requirements
- Python 3.x
- Flask

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hr-chatbot-backend.git
   cd hr-chatbot-backend

Explanation of Code:
Flask App: The backend is built using the Flask web framework. The server listens for POST requests on the /query endpoint.
HR Responses: A simple dictionary hr_responses maps predefined questions to answers.

Query Handling: When a POST request is made with a query, the function get_hr_response looks for the query in the hr_responses dictionary and returns the matching response.

Error Handling: If the query is not recognized, the chatbot responds with a default message asking for clarification.

This is a simple example, and you can expand it with more advanced features such as machine learning, natural language processing (NLP) for more dynamic responses or database integration to store responses and queries.
