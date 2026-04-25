import json

notebook_path = r'C:\ANU\DataMining\InsureDx\notebooks\Full_Project.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
conclusion_cell = None
testing_markdown_added = False

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and cell.get('source'):
        src_text = "".join(cell['source'])
        
        # Split Section 7 and 8 if they are in the same cell
        if '# 7. System Implementation' in src_text and '# 8. Conclusion' in src_text:
            parts = src_text.split('# 8. Conclusion')
            
            # Add Section 7
            new_cells.append({
                'cell_type': 'markdown',
                'metadata': {},
                'source': [parts[0].strip() + '\n\n']
            })
            
            # Save Section 8 to append at the very end
            conclusion_cell = {
                'cell_type': 'markdown',
                'metadata': {},
                'source': ['# 8. Conclusion\n\n' + parts[1].strip() + '\n']
            }
            continue
            
        # Ignore redundant 7.1 cells
        if '### 7.1 Pipeline Testing' in src_text:
            if not testing_markdown_added:
                new_cells.append(cell)
                testing_markdown_added = True
            continue
            
    # Add all other cells normally
    new_cells.append(cell)

# Finally, append the conclusion cell if we found it
if conclusion_cell:
    new_cells.append(conclusion_cell)

nb['cells'] = new_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook successfully cleaned and reordered!')
