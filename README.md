# Flight Delay Chatbot API

A Retrieval-Augmented Generation (RAG) chatbot API built on flight delay data.  
This API powers a Streamlit interface that allows users to interact conversationally with an intelligent assistant capable of predicting and explaining flight delays.  
The system is deployed on Hugging Face Spaces for public interaction.

## Project Overview

This repository hosts the backend API for the Flight Delay RAG Chatbot system.  
The chatbot combines traditional flight delay prediction models with a retrieval-based knowledge component to provide grounded, data-aware responses.  
The FastAPI server handles user queries from the Streamlit frontend, retrieves relevant flight delay information, and generates intelligent, contextual replies.

## Features

- Retrieval-Augmented Generation (RAG) pipeline built on flight delay datasets  
- Conversational API for flight delay insights and explanations  
- Integration with deployed model / inference endpoint on Hugging Face  
- JSON-based REST API design  
- Error handling, validation, and CORS-enabled for frontend integration  
- Ready for Hugging Face or local deployment  

## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
