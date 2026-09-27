import os
import json
import csv
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

from optimize import run_optimization  # ✅ NEW: direct function import
from explain_schedule import explain_schedule

# Ensure uploads folder exists and mounted for frontend access
os.makedirs("uploads", exist_ok=True)

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for enriched vessels.csv)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Health check
@app.get("/")
def root():
    return {"status": "OK", "message": "LNG optimizer backend is live"}

# Upload CSV files
@app.post("/upload/{filename}")
async def upload_csv(filename: str, file: UploadFile = File(...)):
    save_path = os.path.join("data", filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"status": "success", "file": filename}

# Run optimization and return results + enriched vessel CSV
@app.post("/optimize-and-explain")
def optimize_and_explain():
    try:
        results, enriched_vessels, banners = run_optimization()

        # Save enriched schedule
        with open("uploads/vessels.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=enriched_vessels[0].keys())
            writer.writeheader()
            writer.writerows(enriched_vessels)

        # Also save schedule.json (optional — not fetched by frontend)
        with open("results/schedule_output.json", "w") as f:
            json.dump(results, f, indent=2)

        explanation = explain_schedule(results)
        with open("results/explanation.txt", "w") as f:
            f.write(explanation)

        return {
            "status": "success",
            "schedule": results,
            "banners": banners,
            "gpt_explanation": explanation,
            "explanation": explanation
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Optimization failed: {e}"
        }

# Chat assistant
@app.post("/chat")
async def chat_with_schedule(request: Request):
    data = await request.json()
    user_input = data.get("message")
    print("🛰️ User asked:", user_input)

    try:
        with open("results/schedule_output.json", "r") as f:
            schedule = json.load(f)

        system_prompt = (
            "You are a logistics assistant helping with LNG vessel scheduling. "
            "Use only the provided schedule data. "
            "Refer to 'estimated_revenue' and 'estimated_profit' directly. "
            "If data is missing, state that explicitly."
        )

        summary = "\n".join([
            f"- {r['vessel']} (from {r.get('vessel_location', 'unknown')}) → {r['pickup_port']} → {r['delivery_port']} | "
            f"Ballast: {r.get('ballast_days', '?')} days | Total: {r['estimated_days']} days | "
            f"Revenue: ${r['estimated_revenue']} | Cost: ${r.get('estimated_cost', '?')} | Profit: ${r['estimated_profit']}"
            for r in schedule
        ])

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{summary}\n\nUser question: {user_input}"}
            ],
            temperature=0.4
        )

        return {"answer": response.choices[0].message.content}

    except Exception as e:
        print("❌ GPT error:", e)
        return {"error": "Failed to respond from assistant"}