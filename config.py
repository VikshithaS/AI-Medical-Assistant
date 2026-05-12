
"""
SIMPLIFIED CONFIG FILE (NO API KEYS NEEDED)
Multi-Agent Medical Chatbot - Demo Version
"""

import os
from dotenv import load_dotenv

# Load environment variables (optional)
load_dotenv()

# ✅ Dummy LLM (REPLACES ALL REAL MODELS)
class DummyLLM:
    def invoke(self, prompt):
        return "Possible Disease: Viral Infection\nExplanation: This is a demo response based on given symptoms."

    def __call__(self, prompt):
        return "Possible Disease: Viral Infection\nExplanation: Demo medical response."

# ================= CONFIG CLASSES ================= #

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = DummyLLM()

class ConversationConfig:
    def __init__(self):
        self.llm = DummyLLM()

class WebSearchConfig:
    def __init__(self):
        self.llm = DummyLLM()
        self.context_limit = 20

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "none"
        self.embedding_dim = 0
        self.distance_metric = "Cosine"
        self.use_local = False
        self.vector_local_path = "./data/qdrant_db"
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"

        # ❌ Disable embeddings
        self.embedding_model = None

        # ❌ Replace all LLMs
        self.llm = DummyLLM()
        self.summarizer_model = DummyLLM()
        self.chunker_model = DummyLLM()
        self.response_generator_model = DummyLLM()

        self.top_k = 3
        self.vector_search_type = 'similarity'

        self.max_context_length = 1024
        self.include_sources = False
        self.min_retrieval_confidence = 0.0
        self.context_limit = 10

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = ""
        self.chest_xray_model_path = ""
        self.skin_lesion_model_path = ""
        self.skin_lesion_segmentation_output_path = ""
        
        self.llm = DummyLLM()

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = ""
        self.eleven_labs_voice_id = ""

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": False,
            "CHEST_XRAY_AGENT": False,
            "SKIN_LESION_AGENT": False
        }
        self.validation_timeout = 300
        self.default_action = "accept"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5

class UIConfig:
    def __init__(self):
        self.theme = "light"
        self.enable_speech = False
        self.enable_image_upload = False

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.max_conversation_history = 10

