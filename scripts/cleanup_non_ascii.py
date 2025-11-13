#!/usr/bin/env python3
"""Cleanup script to replace non-ASCII/unicode symbols with ASCII equivalents
Applies to .py, .md, .ipynb files under the project directory.
"""
import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTS = {'.py', '.md', '.ipynb'}

# Replacement mapping for common unicode symbols to ASCII
REPLACEMENTS = {
    '\u2014': '--',  # em-dash
    '\u2013': '-',   # en-dash
    '\u2026': '...', # ellipsis
    '->': '->',
    '-': '-',
    '[OK]': '[OK]',
    '[OK]': '[OK]',
    '[X]': '[X]',
    '[i]': '[i]',
    '[i]': '[i]',
    '~': '~',
    '^': '^',
    'x': 'x',
    '--': '--',
    '|': '|',
    '|--': '|--',
    '`--': '`--',
    '-': '-',
    '...': '...',
    '-': '-',
    '->': '->',
    '<-': '<-',
    ' deg': ' deg',
    ' deg': ' deg',
    '"': '"',
    '"': '"',
    ''': "'",
    ''': "'",
    '-': '-',
}

# Emojis: map to empty string
EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]+")

def replace_text(s: str) -> str:
    if not s:
        return s
    # Apply replacements
    for k, v in REPLACEMENTS.items():
        s = s.replace(k, v)
    # Remove remaining emojis
    s = EMOJI_PATTERN.sub('', s)
    # Remove any remaining non-ASCII characters
    s = s.encode('ascii', 'ignore').decode('ascii')
    return s


def process_file(path: Path) -> bool:
    """Return True if modified"""
    changed = False
    if path.suffix == '.ipynb':
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"[SKIP] Could not read JSON: {path} ({e})")
            return False
        modified = False
        for cell in data.get('cells', []):
            if 'source' in cell and cell['source']:
                # source can be list of lines or single string
                if isinstance(cell['source'], list):
                    original = ''.join(cell['source'])
                    replaced = replace_text(original)
                    if replaced != original:
                        # try to preserve line breaks
                        cell['source'] = [line + '\n' for line in replaced.splitlines()]
                        modified = True
                elif isinstance(cell['source'], str):
                    original = cell['source']
                    replaced = replace_text(original)
                    if replaced != original:
                        cell['source'] = replaced
                        modified = True
        if modified:
            path.write_text(json.dumps(data, ensure_ascii=True, indent=1), encoding='utf-8')
            return True
        return False
    else:
        # .py or .md
        try:
            text = path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"[SKIP] Could not read file: {path} ({e})")
            return False
        replaced = replace_text(text)
        if replaced != text:
            path.write_text(replaced, encoding='utf-8')
            return True
        return False


def main():
    modified_files = []
    for p in PROJECT_ROOT.rglob('*'):
        if p.suffix in EXTS:
            ok = process_file(p)
            if ok:
                modified_files.append(str(p.relative_to(PROJECT_ROOT)))
    if modified_files:
        print("Modified files:")
        for f in modified_files:
            print(f"  - {f}")
    else:
        print("No files required modification.")

if __name__ == '__main__':
    main()
