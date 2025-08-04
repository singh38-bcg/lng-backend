import os
import json
import shutil
import subprocess
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins (can restrict to specific URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
def read_root():
    return {"msg": "Hello from FastAPI"}

# Upload endpoint
@app.post("/upload/{filename}")
async def upload_csv(filename: str, file: UploadFile = File(...)):
    file_location = f"data/{filename}"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": f"{filename} uploaded successfully"}

# Optimization + explanation endpoint
@app.post("/optimize-and-explain")
async def optimize_and_explain():
    # Run optimizer
    result = subprocess.run(["python", "optimize.py"], capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "error": "optimize.py failed",
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    # Run explainer
    result = subprocess.run(["python", "explain_schedule.py"], capture_output=True, text=True)
    explanation = result.stdout.split("🧠 GPT Explanation:")[-1].strip()

    # Load output schedule
    if not os.path.exists("results/schedule_output.json"):
        return {
            "Error": "Schedule not found. Please run optimization before asking questions."
        }, 400
    with open("results/schedule_output.json", "r") as f:
        schedule = json.load(f)

    return {
        "schedule": schedule,
        "gpt_explanation": explanation
    }

# AI assistant chat endpoint
@app.post("/chat")
async def chat_with_schedule(request: Request):
    data = await request.json()
    user_input = data.get("message")
    print("🛰️ Received user message:", user_input)

    try:
        with open("results/schedule_output.json", "r") as f:
            schedule = json.load(f)

        system_prompt = (
             "You are a logistics assistant helping with LNG vessel scheduling. "
    "Base your answers strictly on the provided schedule data. "
    "Use the fields 'estimated_revenue' and 'estimated_profit' when calculating margins. "
    "Do not assume or recalculate revenue based on profit. "
    "All values in the schedule are authoritative and accurate. "
    "Respond with clear, numeric, and factual answers only. "
    "If data is missing, say so — do not invent information."
        )

        route_summary = ""
        for item in schedule:
            route_summary += (
                f"- {item['vessel']} is assigned to deliver {item['cargo']} "
                f"from {item['pickup_port']} to {item['delivery_port']} "
                f"(ETA: {item['estimated_days']} days, Profit: ${item['estimated_profit']})\n"
            )

        user_prompt = (
            f"Here is the current LNG vessel schedule:\n\n{route_summary}\n\n"
            f"User question: {user_input}"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
        )
        return {"answer": response.choices[0].message.content}

    except Exception as e:
        print(f"❌ GPT prompt or API error: {e}")
        return {"error": "Failed to generate GPT response due to internal error."}, 500