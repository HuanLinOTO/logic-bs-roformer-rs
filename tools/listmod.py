import sys
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from separate_ref import load_model

m = load_model(Path("assets"))
print([n for n, _ in m.named_children()])
