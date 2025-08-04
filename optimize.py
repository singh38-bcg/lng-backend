import csv
import pulp
import json
from datetime import datetime

# Mapping port to LNG price marker (customizable)
PORT_TO_MARKET = {
    "Yokohama": "JKM",
    "Singapore": "SING",
    "Busan": "JKM",
    "Mumbai": "INDIA",
    "Rotterdam": "TTF"
}

def get_spot_price(destination_port):
    market = PORT_TO_MARKET.get(destination_port, "JKM")
    fake_prices = {
        "JKM": 13.25,
        "SING": 12.80,
        "INDIA": 13.00,
        "TTF": 11.75
    }
    return fake_prices.get(market, 12.00)

def load_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))

vessels = load_csv("data/vessels.csv")
cargos = load_csv("data/cargos.csv")
contracts = load_csv("data/contracts.csv")

# Constants
AVG_DISTANCE = 3000  # nautical miles

# Create LP problem
model = pulp.LpProblem("LNG_Optimization", pulp.LpMaximize)

# Decision variables
assignments = pulp.LpVariable.dicts(
    "assign",
    ((v["vessel_id"], c["cargo_id"]) for v in vessels for c in cargos),
    cat="Binary"
)

# Objective: Maximize profit (revenue – cost)
model += pulp.lpSum([
    assignments[v["vessel_id"], c["cargo_id"]] * (
        float(get_spot_price(c["destination"])) * float(c["volume"])
        - float(v["cost_per_day"]) * (AVG_DISTANCE / float(v["speed"]))
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

# Output
output = []
for v in vessels:
    for c in cargos:
        if pulp.value(assignments[v["vessel_id"], c["cargo_id"]]) == 1:
            speed = float(v["speed"])
            cost_per_day = float(v["cost_per_day"])
            estimated_days = AVG_DISTANCE / speed
            cost = estimated_days * cost_per_day

            spot_price = get_spot_price(c["destination"])
            volume = float(c["volume"])
            revenue = spot_price * volume
            profit = revenue - cost

            output.append({
                "vessel": v["vessel_id"],
                "cargo": c["cargo_id"],
                "pickup_port": c["origin"],
                "delivery_port": c["destination"],
                "estimated_days": round(estimated_days, 2),
                "estimated_revenue": round(revenue, 2),
                "estimated_profit": round(profit, 2),
                "status": "Scheduled",
                "optimized_at": datetime.utcnow().isoformat()
            })

# Save
with open("results/schedule_output.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Optimization complete. Assigned {len(output)} vessels.")
print("📄 Output written to results/schedule_output.json")