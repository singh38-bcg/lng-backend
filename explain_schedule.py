import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load optimized schedule
with open("results/schedule_output.json", "r") as f:
    schedule = json.load(f)

# Format schedule block
summary_block = "**Optimized LNG Vessel Schedule:**\n"
for item in schedule:
    summary_block += (
        f"- {item['vessel']} assigned to {item['cargo']} "
        f"({item['pickup_port']} → {item['delivery_port']}), "
        f"ETA in {item['estimated_days']} days, estimated cost ${item['estimated_cost']}\n"
    )

# Map destinations to vessels
destination_map = {}
for item in schedule:
    dest = item["delivery_port"]
    destination_map.setdefault(dest, []).append(item["vessel"])

# Format destination summary
dest_summary = "**Destination Summary:**\n"
for dest, vessels in destination_map.items():
    dest_summary += f"- {dest}: {', '.join(vessels)}\n"

# GPT system prompt
system_prompt = (
    "You are a logistics assistant. Use only the schedule and destination data provided. "
    "Do not invent vessel statuses, maintenance, KPIs, or statistics. "
    "Answer questions about ports, assignments, and routing strictly based on the provided data."
)

# Combined prompt
user_prompt = f"""
{summary_block}

{dest_summary}

Please explain the schedule and be ready to answer follow-up questions about vessel routing.
"""

# Call OpenAI
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.5
)

# Output response
print("\n🧠 GPT Explanation:\n")
print(response.choices[0].message.content)