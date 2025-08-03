import pandas as pd
import pulp
import json
from datetime import datetime

# Load data
vessels_df = pd.read_csv("data/vessels.csv")
cargos_df = pd.read_csv("data/cargos.csv")
contracts_df = pd.read_csv("data/contracts.csv")

# Constants
AVG_DISTANCE = 3000  # nautical miles, placeholder
HOURS_PER_DAY = 24

# Create a Linear Programming problem
model = pulp.LpProblem("LNG_Lifting_Optimization", pulp.LpMinimize)

# Decision variables: x[vessel][cargo] = 1 if vessel is assigned to cargo
assignments = pulp.LpVariable.dicts(
    "assign",
    ((v, c) for v in vessels_df["vessel_id"] for c in cargos_df["cargo_id"]),
    cat="Binary"
)

# Objective: Minimize cost = (distance / speed) * cost_per_day
model += pulp.lpSum(
    assignments[v, c] *
    (vessels_df[vessels_df["vessel_id"] == v]["cost_per_day"].values[0]) *
    (AVG_DISTANCE / vessels_df[vessels_df["vessel_id"] == v]["speed"].values[0])
    for v in vessels_df["vessel_id"]
    for c in cargos_df["cargo_id"]
)

# Constraint: Each cargo must be assigned to exactly one vessel
for c in cargos_df["cargo_id"]:
    model += pulp.lpSum(assignments[v, c] for v in vessels_df["vessel_id"]) == 1

# Constraint: Each vessel can only be assigned to at most one cargo
for v in vessels_df["vessel_id"]:
    model += pulp.lpSum(assignments[v, c] for c in cargos_df["cargo_id"]) <= 1

# Solve the model
model.solve()

# Extract and format output
output = []
for v in vessels_df["vessel_id"]:
    for c in cargos_df["cargo_id"]:
        if pulp.value(assignments[v, c]) == 1:
            speed = vessels_df[vessels_df["vessel_id"] == v]["speed"].values[0]
            cost_per_day = vessels_df[vessels_df["vessel_id"] == v]["cost_per_day"].values[0]
            estimated_days = AVG_DISTANCE / speed
            estimated_cost = round(cost_per_day * estimated_days, 2)

            output.append({
                "vessel": v,
                "cargo": c,
                "pickup_port": cargos_df[cargos_df["cargo_id"] == c]["origin"].values[0],
                "delivery_port": cargos_df[cargos_df["cargo_id"] == c]["destination"].values[0],
                "estimated_days": round(estimated_days, 2),
                "estimated_cost": estimated_cost,
                "status": "Scheduled",
                "optimized_at": datetime.utcnow().isoformat()
            })

# Save result to JSON
with open("results/schedule_output.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Optimization complete. Assigned {len(output)} vessels.")
print(f"📄 Output written to results/schedule_output.json")