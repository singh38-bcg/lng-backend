import csv
import pulp
import json
from datetime import datetime

# Spot price mapping
import yfinance as yf  # Make sure this is at the top

PORT_TO_MARKET = {
    "Yokohama": "JKM",
    "Singapore": "SING",
    "Busan": "JKM",
    "Mumbai": "INDIA",
    "Rotterdam": "TTF"
}

MARKET_TO_TICKER = {
    "JKM": "JKM=F",
    "TTF": "TTF=F",
    "INDIA": "NG=F",
    "SING": "NG=F"  # Fallback to Henry Hub proxy
}

def get_spot_price(destination_port):
    market = PORT_TO_MARKET.get(destination_port, "JKM")
    ticker_symbol = MARKET_TO_TICKER.get(market)

    try:
        if ticker_symbol:
            ticker = yf.Ticker(ticker_symbol)
            live_price = ticker.fast_info["lastPrice"]
            return round(float(live_price), 2)
    except Exception as e:
        print(f"⚠️ Spot price fetch error for {market}: {e}")

    # Fallback to static
    return {
        "JKM": 13.25,
        "SING": 12.80,
        "INDIA": 13.00,
        "TTF": 11.75
    }.get(market, 12.00)

def load_csv(path, required_keys):
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
        clean = [
            row for row in rows
            if row and all(k in row and row[k].strip() != "" for k in required_keys)
        ]
        print(f"✅ Loaded {len(clean)} valid rows from {path}")
        return clean

# Load and validate data
vessels = load_csv("data/vessels.csv", ["vessel_id", "speed", "cost_per_day"])
cargos = load_csv("data/cargos.csv", ["cargo_id", "origin", "destination", "volume"])
contracts = load_csv("data/contracts.csv", ["cargo_id", "delivery_price_per_ton", "penalty_per_day"])

AVG_DISTANCE = 3000  # nm placeholder

model = pulp.LpProblem("LNG_Lifting_Optimization", pulp.LpMaximize)

assignments = pulp.LpVariable.dicts(
    "assign",
    ((v["vessel_id"], c["cargo_id"]) for v in vessels for c in cargos),
    cat="Binary"
)

# Objective: maximize profit
model += pulp.lpSum([
    assignments[v["vessel_id"], c["cargo_id"]] * (
        get_spot_price(c["destination"]) * float(c["volume"])
        - float(v["cost_per_day"]) * (AVG_DISTANCE / float(v["speed"]))
    )
    for v in vessels for c in cargos
])

# Constraints
for c in cargos:
    model += pulp.lpSum(assignments[v["vessel_id"], c["cargo_id"]] for v in vessels) == 1

for v in vessels:
    model += pulp.lpSum(assignments[v["vessel_id"], c["cargo_id"]] for c in cargos) <= 1

model.solve()

output = []
for v in vessels:
    for c in cargos:
        if pulp.value(assignments[v["vessel_id"], c["cargo_id"]]) == 1:
            eta_days = AVG_DISTANCE / float(v["speed"])
            profit = round(
                get_spot_price(c["destination"]) * float(c["volume"])
                - float(v["cost_per_day"]) * eta_days, 2
            )
            output.append({
                "vessel": v["vessel_id"],
                "cargo": c["cargo_id"],
                "pickup_port": c["origin"],
                "delivery_port": c["destination"],
                "estimated_days": round(eta_days, 2),
                "estimated_profit": profit,
                "status": "Scheduled",
                "optimized_at": datetime.utcnow().isoformat()
            })

# Save result
with open("results/schedule_output.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Optimization complete. Assigned {len(output)} vessels.")
print("📄 Output written to results/schedule_output.json")