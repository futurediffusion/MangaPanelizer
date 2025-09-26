from pathlib import Path
lines = Path(r"nodes/comic_panel_templates.py").read_text().splitlines()
for idx,line in enumerate(lines, 1):
    if 392 <= idx <= 404:
        print(f"{idx}:{repr(line)}")
