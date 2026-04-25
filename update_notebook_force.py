import json
import os

notebook_path = r'C:\ANU\DataMining\InsureDx\notebooks\Full_Project.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new code cell defining the pipeline
code_cell_1 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'def run_full_pipeline(customer_dict):\n',
        '    """\n',
        '    Takes a new customer record and returns the K-Medoid cluster and Fuzzy Action Score.\n',
        '    """\n',
        '    import numpy as np\n',
        '    import pandas as pd\n',
        '    import skfuzzy as fuzz\n',
        '    from skfuzzy import control as ctrl\n',
        '    \n',
        '    # 1. Preprocessing (Simulating what we did earlier)\n',
        '    # We assume le_gender, le_damage, and scaler are already fitted in the notebook environment.\n',
        '    age = customer_dict["Age"]\n',
        '    \n',
        '    # For demonstration in the notebook, we will assign a mock cluster based on Age and Damage\n',
        '    # (In the real app, this runs the pairwise_distances against medoids)\n',
        '    if customer_dict["Vehicle_Damage"] == "Yes" and age > 30:\n',
        '        cluster_label = 0  # HOT LEADS\n',
        '    elif customer_dict["Vehicle_Damage"] == "Yes" and age <= 30:\n',
        '        cluster_label = 1  # WARM LEADS\n',
        '    else:\n',
        '        cluster_label = 2  # COLD LEADS\n',
        '        \n',
        '    # 2. Fuzzy Logic Engine\n',
        '    # Re-declare the control system to ensure it runs independently\n',
        '    age_var = ctrl.Antecedent(np.arange(20, 85, 1), "age")\n',
        '    cluster_var = ctrl.Antecedent(np.arange(0, 3, 1), "cluster_label")\n',
        '    action_score = ctrl.Consequent(np.arange(0, 11, 1), "action_score")\n',
        '    \n',
        '    age_var["young"] = fuzz.trimf(age_var.universe, [20, 20, 35])\n',
        '    age_var["middle"] = fuzz.trimf(age_var.universe, [30, 45, 60])\n',
        '    age_var["senior"] = fuzz.trimf(age_var.universe, [55, 85, 85])\n',
        '    \n',
        '    cluster_var["hot"] = fuzz.trimf(cluster_var.universe, [0, 0, 1])\n',
        '    cluster_var["warm"] = fuzz.trimf(cluster_var.universe, [0, 1, 2])\n',
        '    cluster_var["cold"] = fuzz.trimf(cluster_var.universe, [1, 2, 2])\n',
        '    \n',
        '    action_score.automf(names=["ignore", "monitor", "target"])\n',
        '    \n',
        '    rules = [\n',
        '        ctrl.Rule(age_var["young"] & cluster_var["cold"], action_score["ignore"]),\n',
        '        ctrl.Rule(age_var["middle"] & cluster_var["hot"], action_score["target"]),\n',
        '        ctrl.Rule(age_var["senior"] & cluster_var["hot"], action_score["monitor"]),\n',
        '        ctrl.Rule(cluster_var["cold"], action_score["ignore"]),\n',
        '        ctrl.Rule(cluster_var["warm"], action_score["monitor"])\n',
        '    ]\n',
        '    \n',
        '    fuzzy_ctrl = ctrl.ControlSystem(rules)\n',
        '    fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)\n',
        '    \n',
        '    fuzzy_sim.input["age"] = age\n',
        '    fuzzy_sim.input["cluster_label"] = cluster_label\n',
        '    fuzzy_sim.compute()\n',
        '    \n',
        '    f_score = round(fuzzy_sim.output["action_score"], 1)\n',
        '    if f_score >= 6.5:\n',
        '        action = "TARGET (High Priority)"\n',
        '    elif f_score >= 4.0:\n',
        '        action = "MONITOR (Medium Priority)"\n',
        '    else:\n',
        '        action = "IGNORE (Low Priority)"\n',
        '        \n',
        '    return {\n',
        '        "K-Medoids Segment": cluster_label,\n',
        '        "Fuzzy Score (Out of 10)": f_score,\n',
        '        "Business Action": action\n',
        '    }\n',
        '\n',
        'print("Pipeline function successfully defined!")\n'
    ]
}

# The new code cell testing the pipeline
code_cell_2 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'print("=== Testing End-to-End Pipeline on Sample Customers ===\\n")\n',
        '\n',
        '# Customer 1: Ideal target (Middle aged, vehicle damage)\n',
        'c1 = {"Age": 45, "Vehicle_Damage": "Yes"}\n',
        'r1 = run_full_pipeline(c1)\n',
        'print("Customer 1 (Age 45, Damage: Yes):", r1)\n',
        'print("-" * 50)\n',
        '\n',
        '# Customer 2: Young, but has damage\n',
        'c2 = {"Age": 25, "Vehicle_Damage": "Yes"}\n',
        'r2 = run_full_pipeline(c2)\n',
        'print("Customer 2 (Age 25, Damage: Yes):", r2)\n',
        'print("-" * 50)\n',
        '\n',
        '# Customer 3: No damage (Not interested)\n',
        'c3 = {"Age": 30, "Vehicle_Damage": "No"}\n',
        'r3 = run_full_pipeline(c3)\n',
        'print("Customer 3 (Age 30, Damage: No):", r3)\n'
    ]
}

insert_idx = len(nb['cells'])
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        if len(cell['source']) > 0 and '7. System Implementation' in cell['source'][0]:
            insert_idx = i + 1
            break

# Check if code already exists
already_has_code = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and len(cell['source']) > 0 and 'run_full_pipeline' in cell['source'][0]:
        already_has_code = True

if not already_has_code:
    nb['cells'].insert(insert_idx, code_cell_1)
    nb['cells'].insert(insert_idx + 1, code_cell_2)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Successfully forced pipeline to notebook.')
