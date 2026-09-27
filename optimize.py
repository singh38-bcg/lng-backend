# optimize.py
import csv
import json
import pulp
import shutil
from datetime import datetime, timedelta
import yfinance as yf
from functools import lru_cache

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

@lru_cache(maxsize=None)
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


# ---------------------------------------------------------------------------
# Voyage distances
# ---------------------------------------------------------------------------
# Approximate sea distances in nautical miles between the ports used in the
# data files (Suez routing to/from Europe, Panama to NE Asia from the US Gulf).
# Override or extend by adding data/distances.csv with columns: from,to,nm
PORT_ALIASES = {
    "Qatar Terminal": "Ras Laffan",
    "Port of Rotterdam": "Rotterdam",
    "Singapore Strait": "Singapore",
    "Gulf of Mexico": "Sabine Pass",
}

SEA_DISTANCES_NM = {
    ("Ras Laffan", "Singapore"): 3300,
    ("Ras Laffan", "Yokohama"): 6500,
    ("Ras Laffan", "Busan"): 5900,
    ("Ras Laffan", "Mumbai"): 1100,
    ("Ras Laffan", "Rotterdam"): 6300,
    ("Ras Laffan", "Sabine Pass"): 9800,
    ("Singapore", "Yokohama"): 2900,
    ("Singapore", "Busan"): 2500,
    ("Singapore", "Mumbai"): 2450,
    ("Singapore", "Rotterdam"): 8300,
    ("Singapore", "Sabine Pass"): 11500,
    ("Rotterdam", "Busan"): 10800,
    ("Rotterdam", "Yokohama"): 11200,
    ("Rotterdam", "Mumbai"): 6300,
    ("Rotterdam", "Sabine Pass"): 5000,
    ("Yokohama", "Busan"): 650,
    ("Yokohama", "Mumbai"): 5300,
    ("Yokohama", "Sabine Pass"): 9200,
    ("Busan", "Mumbai"): 4900,
    ("Busan", "Sabine Pass"): 9700,
    ("Mumbai", "Sabine Pass"): 9700,
}

# Used when a vessel's position is unknown (e.g. "Available") or a port pair
# is missing from the table, so the vessel is not treated as already on site.
DEFAULT_POSITIONING_NM = 3000
DEFAULT_VOYAGE_NM = 5000


def _load_distance_overrides(path="data/distances.csv"):
    try:
        with open(path, "r") as f:
            for r in csv.DictReader(f):
                a, b, nm = r.get("from", "").strip(), r.get("to", "").strip(), r.get("nm", "").strip()
                if a and b and nm:
                    SEA_DISTANCES_NM[(normalize_port(a), normalize_port(b))] = float(nm)
    except FileNotFoundError:
        pass


def normalize_port(name):
    name = (name or "").strip()
    return PORT_ALIASES.get(name, name)


def sea_distance_nm(a, b, default):
    """Returns (nautical miles, is_estimate)."""
    a, b = normalize_port(a), normalize_port(b)
    if a and a == b:
        return 0.0, False
    nm = SEA_DISTANCES_NM.get((a, b)) or SEA_DISTANCES_NM.get((b, a))
    if nm is None:
        return float(default), True
    return float(nm), False


def sailing_days(distance_nm, speed_knots):
    # Knots are nautical miles per HOUR, so a day's sailing is speed * 24.
    return distance_nm / (float(speed_knots) * 24.0)


def voyage_plan(vessel, cargo):
    """Ballast leg (current position -> pickup) plus laden leg (pickup -> delivery)."""
    speed = float(vessel["speed"])
    ballast_nm, ballast_est = sea_distance_nm(vessel.get("current_location"), cargo["origin"], DEFAULT_POSITIONING_NM)
    laden_nm, laden_est = sea_distance_nm(cargo["origin"], cargo["destination"], DEFAULT_VOYAGE_NM)
    ballast_days = sailing_days(ballast_nm, speed)
    laden_days = sailing_days(laden_nm, speed)
    total_days = ballast_days + laden_days
    revenue = get_spot_price(cargo["destination"]) * float(cargo["volume"])
    cost = float(vessel["cost_per_day"]) * total_days
    return {
        "ballast_nm": ballast_nm,
        "laden_nm": laden_nm,
        "ballast_days": ballast_days,
        "laden_days": laden_days,
        "total_days": total_days,
        "revenue": revenue,
        "voyage_cost": cost,
        "profit": revenue - cost,
        "distance_estimated": ballast_est or laden_est,
    }


def load_csv(path, required_keys):
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
        return [r for r in rows if all(k in r and r[k].strip() != '' for k in required_keys)]

def run_optimization():
    # delay_hours / last_update are outputs of this optimizer, not required inputs
    vessels = load_csv("data/vessels.csv", ["vessel_id", "speed", "cost_per_day", "current_location", "status"])
    cargos = load_csv("data/cargos.csv", ["cargo_id", "origin", "destination", "window_start", "window_end", "volume"])
    contracts = load_csv("data/contracts.csv", ["cargo_id", "delivery_price_per_ton", "penalty_per_day"])
    _load_distance_overrides()

    if not vessels or not cargos:
        raise ValueError("No usable vessels or cargos found in the uploaded data.")
    if len(cargos) > len(vessels):
        raise ValueError(f"{len(cargos)} cargos but only {len(vessels)} vessels — each vessel can lift one cargo.")

    plans = {(v["vessel_id"], c["cargo_id"]): voyage_plan(v, c) for v in vessels for c in cargos}

    model = pulp.LpProblem("LNG_Lifting_Optimization", pulp.LpMaximize)
    assignments = pulp.LpVariable.dicts("assign", ((v["vessel_id"], c["cargo_id"]) for v in vessels for c in cargos), cat="Binary")

    model += pulp.lpSum([
        assignments[v["vessel_id"], c["cargo_id"]] * plans[v["vessel_id"], c["cargo_id"]]["profit"]
        for v in vessels for c in cargos
    ])

    for c in cargos:
        model += pulp.lpSum(assignments[v["vessel_id"], c["cargo_id"]] for v in vessels) == 1

    for v in vessels:
        model += pulp.lpSum(assignments[v["vessel_id"], c["cargo_id"]] for c in cargos) <= 1

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[model.status] != "Optimal":
        raise ValueError(f"Optimizer could not find a schedule (status: {pulp.LpStatus[model.status]}).")

    now = datetime.utcnow()
    results = []
    enriched = {v["vessel_id"]: v.copy() for v in vessels}
    for v in vessels:
        for c in cargos:
            if pulp.value(assignments[v["vessel_id"], c["cargo_id"]]) > 0.5:
                plan = plans[v["vessel_id"], c["cargo_id"]]
                eta_dt = now + timedelta(days=plan["total_days"])
                eta_str = eta_dt.strftime("%Y-%m-%d %H:%M")
                pickup_eta_dt = now + timedelta(days=plan["ballast_days"])

                window_end_str = c.get("window_end", "")
                try:
                    window_end_dt = datetime.fromisoformat(window_end_str.replace("Z", ""))
                    delay_hours = round((window_end_dt - eta_dt).total_seconds() / 3600)
                except Exception:
                    delay_hours = 0

                enriched[v["vessel_id"]].update({
                    "assignedCargo": c["cargo_id"],
                    "eta": eta_str,
                    "delay_hours": delay_hours,
                    "last_update": now.isoformat()
                })

                results.append({
                    "vessel": v["vessel_id"],
                    "cargo": c["cargo_id"],
                    "vessel_location": v.get("current_location", ""),
                    "pickup_port": c["origin"],
                    "delivery_port": c["destination"],
                    "ballast_nm": round(plan["ballast_nm"]),
                    "laden_nm": round(plan["laden_nm"]),
                    "ballast_days": round(plan["ballast_days"], 2),
                    "laden_days": round(plan["laden_days"], 2),
                    "estimated_days": round(plan["total_days"], 2),
                    "pickup_eta": pickup_eta_dt.strftime("%Y-%m-%d %H:%M"),
                    "estimated_revenue": round(plan["revenue"], 2),
                    "estimated_cost": round(plan["voyage_cost"], 2),
                    "estimated_profit": round(plan["profit"], 2),
                    "distance_estimated": plan["distance_estimated"],
                    "status": "Scheduled",
                    "optimized_at": now.isoformat()
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
        if r.get("delivery_port") == "Busan" and r.get("estimated_days", 0) > 20:
            banners.append({
                "type": "warning",
                "message": "Weather disruption forecasted along route to Busan. Click to explore reroutes."
            })
            break

    return banners

if __name__ == "__main__":
    results, enriched, banners = run_optimization()

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
        print("📣 Banners:", json.dumps(banners, indent=2))
    shutil.copyfile("uploads/vessels.csv", "data/vessels.csv")
    print("✅ Optimization completed and written to files.")