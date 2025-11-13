import json
from pathlib import Path
import sys

def clear_notebook_outputs(notebook_path):
    """Clears all outputs and execution counts from a Jupyter Notebook."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        for cell in nb.get('cells', []):
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error processing {notebook_path}: {e}", file=sys.stderr)
        return False

def main():
    """Finds and clears all notebooks in the project."""
    project_root = Path(__file__).resolve().parents[1]
    notebooks_dir = project_root / 'notebooks'
    
    if not notebooks_dir.exists():
        print(f"Directory not found: {notebooks_dir}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Scanning for notebooks in: {notebooks_dir}")
    cleared_files = []
    
    for notebook_file in notebooks_dir.rglob('*.ipynb'):
        print(f"  - Clearing {notebook_file.name}...")
        if clear_notebook_outputs(notebook_file):
            cleared_files.append(notebook_file.name)
            
    if cleared_files:
        print("\nSuccessfully cleared outputs from:")
        for name in cleared_files:
            print(f"  - {name}")
    else:
        print("\nNo notebooks were found or cleared.")

if __name__ == "__main__":
    main()
