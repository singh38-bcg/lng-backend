import csv
import pulp
import json
from datetime import datetime
import requests

# Mapping port to LNG price marker (customizable)
PORT_TO_MARKET = {
    "Yokohama": "JKM",        # Japan Korea Marker
    "Singapore": "SING",      # Simulated name
    "Busan": "JKM",
    "Mumbai": "INDIA",        # Simulated
    "Rotterdam": "TTF"
}

def get_spot_price(destination_port):
    market = PORT_TO_MARKET.get(destination_port, "JKM")  # fallback
    fake_prices = {
        "JKM": 13.25,
        "SING": 12.80,
        "INDIA": 13.00,
        "TTF": 11.75
    }
    return fake_prices.get(market, 12.00)

# Load CSV data
def load_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))

vessels = load_csv("data/vessels.csv")
cargos = load_csv("data/cargos.csv")

# Constants
AVG_DISTANCE = 3000  # nautical miles, placeholder

# Create a Linear Programming problem
model = pulp.LpProblem("LNG_Lifting_Optimization", pulp.LpMaximize)

# Decision variables
assignments = pulp.LpVariable.dicts(
    "assign",
    ((v["vessel_id"], c["cargo_id"]) for v in vessels for c in cargos),
    cat="Binary"
)

# Objective: Maximize profit
model += pulp.lpSum([
    assignments[v["vessel_id"], c["cargo_id"]] * (
        get_spot_price(c["destination"]) * float(c["volume"]) -
        float(v["cost_per_day"]) * (AVG_DISTANCE / float(v["speed"]))
    )
    for v in vessels for c in cargos
])

# Constraints
for c in cargos:
    model += pulp.lpSum(assignments[v["vessel_id"], c["cargo_id"]] for v in vessels) == 1

for v in vessels:
    model += pulp.lpSum(assignments[v["vessel_id"], c["cargo_id"]] for c in cargos) <= 1

# Solve
model.solve()

# Extract results
output = []
for v in vessels:
    for c in cargos:
        key = (v["vessel_id"], c["cargo_id"])
        if pulp.value(assignments[key]) == 1:
            speed = float(v["speed"])
            cost_per_day = float(v["cost_per_day"])
            days = AVG_DISTANCE / speed
            profit = round(
                get_spot_price(c["destination"]) * float(c["volume"]) -
                cost_per_day * days, 2
            )
            output.append({
                "vessel": v["vessel_id"],
                "cargo": c["cargo_id"],
                "pickup_port": c["origin"],
                "delivery_port": c["destination"],
                "estimated_days": round(days, 2),
                "estimated_profit": profit,
                "status": "Scheduled",
                "optimized_at": datetime.utcnow().isoformat()
            })

# Save to JSON
with open("results/schedule_output.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Optimization complete. Assigned {len(output)} vessels.")
print("📄 Output written to results/schedule_output.json")