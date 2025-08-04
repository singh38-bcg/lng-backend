import csv
import pulp
import json
from datetime import datetime, timedelta
import yfinance as yf

# Mapping
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
    "SING": "NG=F"
}

def get_spot_price(destination_port):
    market = PORT_TO_MARKET.get(destination_port, "JKM")
    ticker_symbol = MARKET_TO_TICKER.get(market)
    try:
        if ticker_symbol:
            ticker = yf.Ticker(ticker_symbol)
            return round(float(ticker.fast_info["lastPrice"]), 2)
    except Exception as e:
        print(f"⚠️ Spot price fetch error for {market}: {e}")
    return {
        "JKM": 13.25, "SING": 12.80, "INDIA": 13.00, "TTF": 11.75
    }.get(market, 12.00)

def load_csv(path, required_keys):
    with open(path, "r") as f:
        reader = list(csv.DictReader(f))
        clean = [r for r in reader if all(k in r and r[k].strip() != "" for k in required_keys)]
        print(f"✅ Loaded {len(clean)} valid rows from {path}")
        return clean

# Load CSVs
vessels = load_csv("data/vessels.csv", ["vessel_id", "speed", "cost_per_day"])
cargos = load_csv("data/cargos.csv", ["cargo_id", "origin", "destination", "window_end", "volume"])
contracts = load_csv("data/contracts.csv", ["cargo_id", "delivery_price_per_ton", "penalty_per_day"])

AVG_DISTANCE = 3000  # placeholder nautical miles

# Build LP
model = pulp.LpProblem("LNG_Optimization", pulp.LpMaximize)

assignments = pulp.LpVariable.dicts(
    "assign",
    ((v["vessel_id"], c["cargo_id"]) for v in vessels for c in cargos),
    cat="Binary"
)

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

model.solve()

# Build outputs
results = []
updated_vessels = {v["vessel_id"]: v.copy() for v in vessels}

for v in vessels:
    for c in cargos:
        if pulp.value(assignments[v["vessel_id"], c["cargo_id"]]) == 1:
            eta_days = AVG_DISTANCE / float(v["speed"])
            eta_date = datetime.utcnow() + timedelta(days=eta_days)
            eta_str = eta_date.strftime("%Y-%m-%d %H:%M")
            window_end = datetime.fromisoformat(c["window_end"].replace("Z", ""))

            delay = (window_end - eta_date).total_seconds() / 3600
            delay_hours = round(delay)

            profit = round(
                get_spot_price(c["destination"]) * float(c["volume"]) -
                float(v["cost_per_day"]) * eta_days, 2
            )

            updated_vessels[v["vessel_id"]].update({
                "assignedCargo": c["cargo_id"],
                "eta": eta_str,
                "delay_hours": delay_hours,
                "last_update": datetime.utcnow().isoformat()
            })

            results.append({
                "vessel": v["vessel_id"],
                "cargo": c["cargo_id"],
                "pickup_port": c["origin"],
                "delivery_port": c["destination"],
                "estimated_days": round(eta_days, 2),
                "estimated_revenue": round(get_spot_price(c["destination"]) * float(c["volume"]), 2),
                "estimated_profit": profit,
                "status": "Scheduled",
                "optimized_at": datetime.utcnow().isoformat()
            })

# Save schedule
with open("results/schedule_output.json", "w") as f:
    json.dump(results, f, indent=2)

# Save updated vessel status
with open("uploads/vessels.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(updated_vessels.values())[0].keys())
    writer.writeheader()
    writer.writerows(updated_vessels.values())

print(f"✅ Optimization complete. {len(results)} assignments saved.")
print("📄 schedule_output.json and updated vessels.csv written.")