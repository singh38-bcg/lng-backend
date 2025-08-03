import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from openai import OpenAI
from fastapi import UploadFile, File
import shutil

# Load .env and API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Health check route
@app.get("/")
def read_root():
    return {"msg": "Hello from FastAPI"}

# Chat endpoint
@app.post("/chat")
async def chat_with_schedule(request: Request):
    data = await request.json()
    user_input = data.get("message")

    # Load the latest optimized schedule
    with open("results/schedule_output.json", "r") as f:
        schedule = json.load(f)

    # Build route summary text
    route_summary = ""
    for item in schedule:
        route_summary += (
            f"- {item['vessel']} is assigned to deliver {item['cargo']} "
            f"from {item['pickup_port']} to {item['delivery_port']} "
            f"(ETA: {item['estimated_days']} days, Cost: ${item['estimated_cost']})\n"
        )

    # Construct GPT prompt
    system_prompt = (
        "You are a logistics assistant helping with LNG vessel scheduling. "
        "Base your answers only on the provided schedule data. "
        "Do not invent fleet stats, maintenance info, or make assumptions."
    )

    user_prompt = (
        f"Here is the current LNG vessel schedule:\n\n{route_summary}\n\n"
        f"User question: {user_input}"
    )

    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4,
    )

    return {"answer": response.choices[0].message.content}
@app.post("/upload/{filename}")
async def upload_csv(filename: str, file: UploadFile = File(...)):
    file_location = f"data/{filename}"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": f"{filename} uploaded successfully"}