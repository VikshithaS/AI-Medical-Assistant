# AI Medical Assistant

AI Medical Assistant is a web-based healthcare support system developed using FastAPI, HTML, CSS, and JavaScript. The project provides preliminary medical guidance through chat, voice interaction, and medical report analysis.

## Features

- Symptom-based health guidance
- Voice input support using speech recognition
- Multi-step doctor-like conversation flow
- Medicine suggestions and care instructions
- Medical report and image upload
- AI-inspired multi-agent architecture
- Interactive chatbot interface

## Technologies Used

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- FastAPI
- Python

### Database
- Qdrant Vector Database

### AI Modules
- Conversational Agent
- RAG Agent
- Web Search Agent
- Image Analysis Agent

## System Workflow

User → UI → FastAPI Backend → Agent Decision Module → Processing → Response

## Project Architecture

- Frontend handles user interaction through chat, voice, and uploads.
- Backend processes requests and routes them to appropriate agents.
- Agents analyze symptoms and generate responses.
- Qdrant database stores and retrieves medical embeddings.

## How to Run the Project

### Install Requirements

```bash
pip install -r requirements.txt
## To run the server
python -m uvicorn app:app --reload
