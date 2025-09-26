from pathlib import Path
text = Path(r"nodes/comic_panel_templates.py").read_text()
marker = text.index("            elif first_char == \"V\":\n                # Similar logic for vertical layouts with enhanced diagonals")
end = text.index("            else:\n                draw.text", marker)
old = text[marker:end]
print(old)
