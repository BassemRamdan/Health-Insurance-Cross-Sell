import json

notebook_path = r'C:\ANU\DataMining\InsureDx\notebooks\Full_Project.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell that has "=== Testing End-to-End Pipeline"
test_code_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and len(cell['source']) > 0 and 'Testing End-to-End Pipeline' in cell['source'][0]:
        test_code_idx = i
        break

if test_code_idx != -1:
    md_cell = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '### 7.1 Pipeline Testing & Validation\n',
            'To prove the robustness of our Fuzzy-Clustering pipeline, we test the `run_full_pipeline` function on three distinct customer profiles:\n',
            '\n',
            '- **Customer 1 (The Ideal Lead):** Middle-aged (45) with a history of vehicle damage. We expect the system to categorize them as a Hot Lead and output a high **TARGET** action score.\n',
            '- **Customer 2 (The Borderline Lead):** Young (25) but has a history of vehicle damage. We expect the system to categorize them as a Warm Lead and output a **MONITOR** action score.\n',
            '- **Customer 3 (The Cold Lead):** No history of vehicle damage. We expect the system to immediately categorize them as a Cold Lead and output an **IGNORE** action score to save marketing costs.'
        ]
    }
    nb['cells'].insert(test_code_idx, md_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print('Markdown added successfully!')
else:
    print('Testing code cell not found!')
