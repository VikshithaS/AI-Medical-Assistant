import os
import uuid
import tempfile
from typing import Dict, Union, Optional, List
import glob
import threading
import time
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import uvicorn
import requests
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs

from config import Config
#from agents.agent_decision import process_query
session_memory = {}

def process_query(query, session_id="default"):
    query = query.lower()

    # ---------- GREETING ----------
    if any(word in query for word in ["hi", "hello", "hey"]):
        return {
            "messages": [
                type("obj",(object,),{
                    "content": "Hello! I am your AI Medical Assistant.\n\nPlease describe your symptoms so I can help you."
                })()
            ]
        }

    # ---------- THANK YOU ----------
    if any(word in query for word in ["thanks", "thank you"]):
        return {
            "messages": [
                type("obj",(object,),{
                    "content": "You're welcome. Stay healthy!"
                })()
            ]
        }

    # ---------- INIT SESSION ----------
    if session_id not in session_memory:
        session_memory[session_id] = {"stage": 0, "data": {}}

    user = session_memory[session_id]

    # ---------- STAGE 0 ----------
    if user["stage"] == 0:
        user["data"]["symptom"] = query
        user["stage"] = 1

        return {"messages":[type("obj",(object,),{
            "content":"How many days have you been experiencing this?"
        })()]}

    # ---------- STAGE 1 ----------
    elif user["stage"] == 1:
        user["data"]["duration"] = query
        user["stage"] = 2

        return {"messages":[type("obj",(object,),{
            "content":"Is the severity mild, moderate, or severe?"
        })()]}

    # ---------- STAGE 2 ----------
    elif user["stage"] == 2:
        user["data"]["severity"] = query
        user["stage"] = 3

        return {"messages":[type("obj",(object,),{
            "content":"Do you have additional symptoms like fever, cold, vomiting, headache, or body pain?"
        })()]}

    # ---------- FINAL DIAGNOSIS ----------
    elif user["stage"] == 3:

        extra = query
        symptom = user["data"]["symptom"]
        duration = user["data"]["duration"]
        severity = user["data"]["severity"]

        condition = "General Illness"
        medicines = []
        advice = []
        warnings = []
        emergency = []

        # -------- FEVER / VIRAL --------
        if "fever" in symptom or "fever" in extra:
            condition = "Viral Fever / Infection"

            medicines = [
                "Paracetamol (500mg for fever)",
                "Cough syrup (if cough present)",
                "Antihistamine (for cold)"
            ]

            advice = [
                "Drink warm fluids and stay hydrated",
                "Take adequate rest",
                "Use light and nutritious food",
                "Avoid cold drinks and junk food",
                "Take steam inhalation if congestion present"
            ]

            warnings = [
                "Fever lasting more than 3 days",
                "High temperature above 102°F",
                "Persistent cough or weakness"
            ]

            emergency = [
                "Difficulty in breathing",
                "Severe chest pain",
                "Unconsciousness"
            ]

        # -------- COLD --------
        elif "cold" in symptom or "cough" in symptom or "cold" in extra:
            condition = "Common Cold"

            medicines = [
                "Antihistamine tablets",
                "Steam inhalation",
                "Decongestant syrup"
            ]

            advice = [
                "Drink warm water",
                "Take steam twice daily",
                "Avoid cold food items",
                "Rest properly",
                "Use saline nasal drops if needed"
            ]

            warnings = [
                "Cold lasting more than a week",
                "Severe throat pain",
                "Persistent nasal blockage"
            ]

            emergency = [
                "Breathing difficulty",
                "High fever with cold"
            ]

        # -------- VOMITING --------
        elif "vomit" in symptom or "vomiting" in extra:
            condition = "Gastric Issue / Food Poisoning"

            medicines = [
                "ORS solution",
                "Antacid (Gelusil)",
                "Domperidone (for vomiting control)"
            ]

            advice = [
                "Stay hydrated (ORS/water)",
                "Avoid oily and spicy food",
                "Eat light meals (rice, curd)",
                "Take small frequent meals",
                "Rest properly"
            ]

            warnings = [
                "Continuous vomiting",
                "Signs of dehydration",
                "Severe stomach pain"
            ]

            emergency = [
                "Blood in vomit",
                "Severe dehydration",
                "Unable to drink fluids"
            ]

        # -------- GENERAL --------
        else:
            condition = "General Weakness / Stress"

            medicines = [
                "Paracetamol",
                "Vitamin supplements"
            ]

            advice = [
                "Take rest",
                "Stay hydrated",
                "Maintain proper sleep",
                "Reduce screen time"
            ]

            warnings = [
                "Symptoms persisting more than 2–3 days"
            ]

        # RESET SESSION
        session_memory[session_id] = {"stage": 0, "data": {}}

        # ---------- FINAL RESPONSE ----------
        response = rf"""
Final Assessment: {condition}

Based on:
• Symptom: {symptom}
• Duration: {duration}
• Severity: {severity}
• Extra Symptoms: {extra}

Suggested Medicines:
{chr(10).join("• " + m for m in medicines)}

Care Instructions:
{chr(10).join("• " + a for a in advice)}

When to See a Doctor:
{chr(10).join("• " + w for w in warnings)}

Emergency Situations:
{chr(10).join("• " + e for e in emergency)}

Important Note:
This is preliminary AI-based guidance. Please consult a healthcare professional for proper diagnosis.
"""

        return {
            "messages":[type("obj",(object,),{"content":response})()]
        }
# Load configuration
config = Config()

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Medical Chatbot", version="2.0")

# Set up directories
UPLOAD_FOLDER = "uploads/backend"
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"
SPEECH_DIR = "uploads/speech"

# Create directories if they don't exist
for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SKIN_LESION_OUTPUT, SPEECH_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mount static files directory
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Initialize ElevenLabs client
client = ElevenLabs(
    api_key=config.speech.eleven_labs_api_key,
)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_audio():
    """Deletes all .mp3 files in the uploads/speech folder every 5 minutes."""
    while True:
        try:
            files = glob.glob(f"{SPEECH_DIR}/*.mp3")
            for file in files:
                os.remove(file)
            print("Cleaned up old speech files.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        time.sleep(300)  # Runs every 5 minutes

# Start background cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
cleanup_thread.start()

class QueryRequest(BaseModel):
    query: str
    conversation_history: List = []

class SpeechRequest(BaseModel):
    text: str
    voice_id: str = "EXAMPLE_VOICE_ID"  # Default voice ID

@app.get("/", response_class=HTMLResponse)
def home():
    return r"""
<html>
<head>
<title>Medical Chatbot</title>
</head>

<body style="margin:0; font-family: Arial; background:#121212; color:white; display:flex; flex-direction:column; height:100vh;">

<div style="padding:15px; text-align:center; position:relative;">
    🩺 Medical Chatbot

    <button onclick="toggleVoice()" 
        style="position:absolute; right:20px; top:10px;">
        🔊
    </button>
</div>

<div id="chatbox" style="flex:1; padding:20px; overflow-y:auto;"></div>

<div style="display:flex; padding:10px; background:#1e1e1e;">

  <label style="margin-right:5px; cursor:pointer; font-size:20px;">
    ➕
    <input type="file" id="fileInput" style="display:none;">
</label>

    <input id="userInput" placeholder="Type symptoms..."
        style="flex:1; padding:12px; border:none; border-radius:5px;">
    
    <button onclick="uploadFile()" style="margin-right:5px;">Upload</button>

    <button onclick="sendMessage()" 
        style="margin-left:5px; padding:12px; background:#007bff; color:white; border:none; border-radius:5px;">
        Send
    </button>
<button onclick="startListening()" 
    style="margin-left:5px; padding:12px; background:#28a745; color:white; border:none; border-radius:5px;">
    🎤
</button>
</div>

<script>

// ================= BASIC SETUP =================
let voiceEnabled = true;

// ================= MIC =================
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.lang = "en-US";

    recognition.onresult = function(event){
        let text = event.results[0][0].transcript;
        document.getElementById("userInput").value = text;
        sendMessage();
    };

} else {
    console.log("Speech recognition not supported");
}

// ================= ENTER =================
document.addEventListener("DOMContentLoaded", function(){
    document.getElementById("userInput").addEventListener("keypress", function(e){
        if(e.key === "Enter") sendMessage();
    });
});

// ================= SEND MESSAGE =================
window.sendMessage = async function(){

    let inputBox = document.getElementById("userInput");
    let input = inputBox.value.trim();
    if(!input) return;

    let chatbox = document.getElementById("chatbox");

    // USER MESSAGE
    chatbox.innerHTML += `
    <div style="text-align:right;margin:10px;">
        <span style="background:#007bff;color:white;padding:10px;border-radius:10px;">
            ${input}
        </span>
    </div>`;

    inputBox.value = "";

    // TYPING
    let id = "t" + Date.now();
    chatbox.innerHTML += `
    <div id="${id}" style="margin:10px;">
        <span style="background:#333;padding:10px;border-radius:10px;">
            Typing...
        </span>
    </div>`;

    chatbox.scrollTop = chatbox.scrollHeight;

    try {

        let res = await fetch("/chat", {
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body: JSON.stringify({query:input})
        });

        let data = await res.json();

        document.getElementById(id).remove();

        // BOT MESSAGE
        chatbox.innerHTML += `
        <div style="margin:10px;">
            <span style="background:#2c2c2c;padding:10px;border-radius:10px;display:inline-block;max-width:60%;">
                ${data.response.replace(/\n/g,"<br>")}
            </span>
        </div>`;

        chatbox.scrollTop = chatbox.scrollHeight;

        // VOICE
        if (voiceEnabled && "speechSynthesis" in window){

            speechSynthesis.cancel();

            let cleanText = data.response.replace(/[^\x00-\x7F]/g, "");

            let msg = new SpeechSynthesisUtterance(cleanText);

            msg.rate = 1;
            msg.pitch = 1;

            speechSynthesis.speak(msg);
        }

    } catch (error) {
        console.log("Error:", error);
    }
};

// ================= MIC BUTTON =================
window.startListening = function(){
    if(recognition){
        try{
            recognition.start();
        }catch(e){
            console.log("Mic error");
        }
    } else {
        alert("Mic not supported");
    }
};

// ================= FILE UPLOAD =================
window.uploadFile = async function(){

    let fileInput = document.getElementById("fileInput");
    let file = fileInput.files[0];

    if(!file){
        alert("Select a file");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    try{
        let res = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        let data = await res.json();

        let chatbox = document.getElementById("chatbox");

        chatbox.innerHTML += `
        <div style="margin:10px;">
            <span style="background:#444;padding:10px;border-radius:10px;">
                ${data.analysis}
            </span>
        </div>`;

    } catch(err){
        console.log("Upload error:", err);
    }
};

// ================= VOICE TOGGLE =================
window.toggleVoice = function(){
    voiceEnabled = !voiceEnabled;
    alert("Voice " + (voiceEnabled ? "ON 🔊" : "OFF 🔇"));
};

</script>

</body>
</html>
"""
@app.get("/health")
def health_check():
    """Health check endpoint for Docker health checks"""
    return {"status": "healthy"}


from fastapi import Request

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query", "")

    session_id = request.client.host  # simple session id

    response_data = process_query(query, session_id)
    response_text = response_data['messages'][-1].content

    return {"response": response_text}
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        response_data = process_query(request.query)
        response_text = response_data['messages'][-1].content
        
        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # Check if the agent is skin lesion segmentation and find the image path
        result = {
            "status": "success",
            "response": response_text, 
            "agent": response_data["agent_name"]
        }
        
        # If it's the skin lesion segmentation agent, check for output image
        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    from fastapi import UploadFile, File

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    if "fever" in text.lower():
        result = "Report indicates possible infection."
    else:
        result = "No major issues detected in report."

    return {"analysis": result}

@app.post("/upload")
async def upload_image(
    response: Response,
    image: UploadFile = File(...), 
    text: str = Form(""),
    session_id: Optional[str] = Cookie(None)
):
    """Process medical image uploads with optional text input."""
    # Validate file type
    if not allowed_file(image.filename):
        return JSONResponse(
            status_code=400, 
            content={
                "status": "error",
                "agent": "System",
                "response": "Unsupported file type. Allowed formats: PNG, JPG, JPEG"
            }
        )
    
    # Check file size before saving
    file_content = await image.read()
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:  # Convert MB to bytes
        return JSONResponse(
            status_code=413, 
            content={
                "status": "error",
                "agent": "System",
                "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
            }
        )
    
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Save file securely
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    try:
        query = {"text": text, "image": file_path}
        response_data = process_query(query)
        response_text = response_data['messages'][-1].content

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # Check if the agent is skin lesion segmentation and find the image path
        result = {
            "status": "success",
            "response": response_text, 
            "agent": response_data["agent_name"]
        }
        
        # If it's the skin lesion segmentation agent, check for output image
        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")
        
        # Remove temporary file after sending
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to remove temporary file: {str(e)}")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
def validate_medical_output(
    response: Response,
    validation_result: str = Form(...), 
    comments: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None)
):
    """Handle human validation for medical AI outputs."""
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)
        
        # Re-run the agent decision system with the validation input
        validation_query = f"Validation result: {validation_result}"
        if comments:
            validation_query += f" Comments: {comments}"
        
        response_data = process_query(validation_query)

        if validation_result.lower() == 'yes':
            return {
                "status": "validated",
                "message": "**Output confirmed by human validator:**",
                "response": response_data['messages'][-1].content
            }
        else:
            return {
                "status": "rejected",
                "comments": comments,
                "message": "**Output requires further review:**",
                "response": response_data['messages'][-1].content
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Endpoint to transcribe speech using ElevenLabs API"""
    if not audio.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "No audio file selected"}
        )
    
    try:
        # Save the audio file temporarily
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.webm"
        
        # Read and save the file
        audio_content = await audio.read()
        with open(temp_audio, "wb") as f:
            f.write(audio_content)
        
        # Debug: Print file size to check if it's empty
        file_size = os.path.getsize(temp_audio)
        print(f"Received audio file size: {file_size} bytes")
        
        if file_size == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Received empty audio file"}
            )
        
        # Convert to MP3
        mp3_path = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.mp3"
        
        try:
            # Use pydub with format detection
            audio = AudioSegment.from_file(temp_audio)
            audio.export(mp3_path, format="mp3")
            
            # Debug: Print MP3 file size
            mp3_size = os.path.getsize(mp3_path)
            print(f"Converted MP3 file size: {mp3_size} bytes")

            with open(mp3_path, "rb") as mp3_file:
                audio_data = mp3_file.read()
            print(f"Converted audio file into byte array successfully!")

            transcription = client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng",
                diarize=True,
            )
            
            # Clean up temp files
            try:
                os.remove(temp_audio)
                os.remove(mp3_path)
                print(f"Deleted temp files: {temp_audio}, {mp3_path}")
            except Exception as e:
                print(f"Could not delete file: {e}")
            
            if transcription.text:
                return {"transcript": transcription.text}
            else:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"API error: {transcription}", "details": transcription.text}
                )

        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing audio: {str(e)}"}
            )
                
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    """Endpoint to generate speech using ElevenLabs API"""
    try:
        text = request.text
        selected_voice_id = request.voice_id
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Text is required"}
            )
        
        # Define API request to ElevenLabs
        elevenlabs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": config.speech.eleven_labs_api_key
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        # Send request to ElevenLabs API
        response = requests.post(elevenlabs_url, headers=headers, json=payload)

        if response.status_code != 200:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to generate speech, status: {response.status_code}", "details": response.text}
            )
        
        # Save the audio file temporarily
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio_path = f"./{SPEECH_DIR}/{uuid.uuid4()}.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(response.content)

        # Return the generated audio file
        return FileResponse(
            path=temp_audio_path,
            media_type="audio/mpeg",
            filename="generated_speech.mp3"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Add exception handler for request entity too large
@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "status": "error",
            "agent": "System",
            "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)