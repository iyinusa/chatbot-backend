# Multilingual Customer Support Chatbot - Backend

Backend service for the Multilingual Conversational Dialogue System: A Case Study of Customer Support Bots.

## Overview

This repository contains the backend implementation for a multilingual customer support chatbot system developed as part of an MSc Computing Dissertation project. The system is designed to handle customer inquiries across multiple languages and provide appropriate responses for various customer service scenarios, particularly focusing on order management and customer support interactions.

## Features

- **Multilingual Support**: Handles conversations in multiple languages including English, Hindi, Urdu, Italian, and more
- **Intent Classification**: Accurately categorises customer queries by intent (e.g., order changes, cancellations)
- **Response Generation**: Provides contextually appropriate responses based on categorized intents
- **BLEU Score Evaluation**: Includes functionality for evaluating response quality using BLEU scoring metrics
- **Comprehensive Dataset**: Utilises a rich dataset of customer inquiries and responses
- **API Endpoints**: RESTful API for seamless frontend integration
- **Training Data Processing**: Tools for processing and managing training datasets

## Dataset

The system utilises:

- A primary dataset (dataset.csv) utilising [HuggingFace Bitext Customer Support LLM Chatbot Training
Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) containing customer inquiries and appropriate responses
- Training data in JSONL format for machine learning model training
- Evaluation metrics via BLEU scores for response quality assessment

## Related Projects

- **Frontend Repository**: [chatbot-frontend](https://github.com/iyinusa/chatbot-front) - The user interface that connects to this backend

## System Architecture

The backend implements a pipeline architecture:
1. **Input Processing**: Handles incoming customer queries
2. **Language Detection**: Identifies the language of the query
3. **Intent Classification**: Determines the customer's intent
4. **Response Generation**: Produces appropriate responses
5. **API Delivery**: Delivers responses to the frontend interface

## Setup and Installation

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/iyinusa/chatbot-backend.git
   cd chatbot-backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```
   OPENAI_KEY=''
   ```

4. Start the development server:
   ```bash
   flask run
   ```

## Frontend Integration

To integrate with the frontend:

1. Ensure the backend server is running
2. Configure the frontend to make API calls to the backend endpoints
3. Set CORS settings in the backend to allow requests from the frontend origin
4. Follow the documentation in the [frontend repository](https://github.com/iyinusa/chatbot-front)

## Troubleshooting

Common issues:
- **CORS issues**: Check CORS settings in the backend configuration
- **Dependency problems**: Make sure all dependencies are properly installed with `pip install -r requirement.txt`

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

I. Kennedy Yinusa
