import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a logistics assistant helping with LNG vessel scheduling. "
    "Base your answers strictly on the provided schedule data. "
    "Use the fields 'estimated_revenue', 'estimated_cost' and 'estimated_profit' when discussing margins. "
    "Explain why each vessel was chosen, referring to its starting position and the ballast "
    "(repositioning) distance to the pickup port. "
    "Respond with clear, numeric, and factual answers only. "
    "If data is missing, say so — do not invent information."
)


def _schedule_lines(schedule):
    lines = []
    for item in schedule:
        lines.append(
            f"- {item['vessel']} (currently at {item.get('vessel_location') or 'unknown'}) assigned to {item['cargo']}: "
            f"{item.get('ballast_nm', '?')} nm ballast to {item['pickup_port']} ({item.get('ballast_days', '?')} days), "
            f"then {item.get('laden_nm', '?')} nm laden to {item['delivery_port']} ({item.get('laden_days', '?')} days). "
            f"Total {item['estimated_days']} days. Revenue ${item['estimated_revenue']:,.0f}, "
            f"voyage cost ${item.get('estimated_cost', 0):,.0f}, profit ${item['estimated_profit']:,.0f}."
            + (" (Some distances are estimates.)" if item.get("distance_estimated") else "")
        )
    return "\n".join(lines)


def fallback_explanation(schedule):
    """Plain-text explanation used when the LLM call is unavailable."""
    if not schedule:
        return "No cargos were scheduled."
    total_profit = sum(r["estimated_profit"] for r in schedule)
    return (
        "The optimizer assigned each cargo to the vessel that maximizes total profit, counting both "
        "the ballast leg to the pickup port and the laden voyage.\n"
        + _schedule_lines(schedule)
        + f"\nTotal estimated profit: ${total_profit:,.0f}."
    )


def explain_schedule(schedule, model="gpt-3.5-turbo"):
    """Returns a written explanation of the schedule. Never raises."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not schedule:
        return fallback_explanation(schedule)
    try:
        client = OpenAI(api_key=api_key)
        user_prompt = (
            "**Optimized LNG Vessel Schedule:**\n"
            + _schedule_lines(schedule)
            + "\n\nPlease explain this schedule concisely: why each vessel was picked and the overall economics."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or fallback_explanation(schedule)
    except Exception as e:
        print("⚠️ Explanation generation failed, using fallback:", e)
        return fallback_explanation(schedule)


if __name__ == "__main__":
    with open("results/schedule_output.json", "r") as f:
        schedule = json.load(f)
    print("\n🧠 GPT Explanation:\n")
    print(explain_schedule(schedule))
