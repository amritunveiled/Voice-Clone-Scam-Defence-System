from pathlib import Path
required=["Factorial design","Audio conditions","Splits","Systems","Metrics","Live metrics","Freeze rule"]
p=Path("PROTOCOL.md")
text=p.read_text(encoding="utf-8")
missing=[x for x in required if x not in text]
if missing:
    raise SystemExit("Missing protocol sections: "+", ".join(missing))
print("PROTOCOL STATUS: PASS")
