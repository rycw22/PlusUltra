import json
from pathlib import Path

in_path = Path("echo_frames_dummy_anno.json")
out_path = Path("echo_frames_dummy_anno_FIXED.json")

data = json.loads(in_path.read_text())

for img in data["images"]:
    fn = img.get("file_name", "")
    # keep only basename if it contains folders
    img["file_name"] = Path(fn).name

out_path.write_text(json.dumps(data))
print("Wrote:", out_path)

