# optimize.py
import csv
import json
import pulp
import shutil
from datetime import datetime, timedelta
import yfinance as yf

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
            price = ticker.fast_info.get("lastPrice") or ticker.info.get("regularMarketPrice")
            if price:
                return round(float(price), 2)
    except Exception as e:
        print(f"⚠️ Spot price fetch error for {market}: {e}")

    # Fallback static prices
    return {
        "JKM": 13.25,
        "SING": 12.80,
        "INDIA": 13.00,
        "TTF": 11.75
    }.get(market, 12.00)

def load_csv(path, required_keys):
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
        return [r for r in rows if all(k in r and r[k].strip() != '' for k in required_keys)]

def run_optimization():
    vessels = load_csv("data/vessels.csv", ["vessel_id", "speed", "cost_per_day", "current_location","status", "delay_hours", "last_update"])
    cargos = load_csv("data/cargos.csv", ["cargo_id", "origin", "destination", "window_start", "window_end", "volume"])
    contracts = load_csv("data/contracts.csv", ["cargo_id", "delivery_price_per_ton", "penalty_per_day"])

    model = pulp.LpProblem("LNG_Lifting_Optimization", pulp.LpMaximize)
    assignments = pulp.LpVariable.dicts("assign", ((v["vessel_id"], c["cargo_id"]) for v in vessels for c in cargos), cat="Binary")

    AVG_DISTANCE = 3000

    model += pulp.lpSum([
        assignments[v["vessel_id"], c["cargo_id"]] * (
            get_spot_price(c["destination"]) * float(c["volume"]) - float(v["cost_per_day"]) * (AVG_DISTANCE / float(v["speed"]))
        )
        for v in vessels for c in cargos
    ])

    for c in cargos:
        model += pulp.lpSum(assignments[v["vessel_id"], c["cargo_id"]] for v in vessels) == 1

    for v in vessels:
        model += pulp.lpSum(assignments[v["vessel_id"], c["cargo_id"]] for c in cargos) <= 1

    model.solve()

    results = []
    enriched = {v["vessel_id"]: v.copy() for v in vessels}
    for v in vessels:
        for c in cargos:
            if pulp.value(assignments[v["vessel_id"], c["cargo_id"]]) == 1:
                eta_days = AVG_DISTANCE / float(v["speed"])
                eta_dt = datetime.utcnow() + timedelta(days=eta_days)
                eta_str = eta_dt.strftime("%Y-%m-%d %H:%M")

                window_end_str = c.get("window_end", "")
                try:
                    window_end_dt = datetime.fromisoformat(window_end_str.replace("Z", ""))
                    delay_hours = round((window_end_dt - eta_dt).total_seconds() / 3600)
                except Exception:
                    delay_hours = 0

                profit = round(get_spot_price(c["destination"]) * float(c["volume"]) - float(v["cost_per_day"]) * eta_days, 2)

                enriched[v["vessel_id"]].update({
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

    return results, list(enriched.values()), generate_banners(results)

def generate_banners(results):
    banners = []

    # Spot price alerts for any delivery port
    seen_ports = set()
    for r in results:
        port = r.get("delivery_port")
        if port and port not in seen_ports:
            spot_price = get_spot_price(port)
            if spot_price >= 13.0:
                banners.append({
                    "type": "opportunity",
                    "message": f"Spot price surge at {port} — consider selling 20kt for ${spot_price:.2f}/mmBtu."
                })
            seen_ports.add(port)

    # Simulated weather alert for Busan
    for r in results:
        if r.get("delivery_port") == "Busan" and r.get("estimated_days", 0) > 140:
            banners.append({
                "type": "warning",
                "message": "Weather disruption forecasted along route to Busan. Click to explore reroutes."
            })
            break

    return banners

if __name__ == "__main__":
    results, enriched = run_optimization()

    # Save results
    with open("results/schedule_output.json", "w") as f:
        json.dump(results, f, indent=2)

    # Safely write enriched vessels CSV
    with open("uploads/vessels.csv", "w", newline="") as f:
        fieldnames = [
            "vessel_id", "speed", "cost_per_day", "current_location", "status",
            "delay_hours", "last_update", "assignedCargo", "eta"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)
        banners = generate_banners(results)
        print("📣 Banners:", json.dumps(banners, indent=2))
    shutil.copyfile("uploads/vessels.csv", "data/vessels.csv")
    print("✅ Optimization completed and written to files.")