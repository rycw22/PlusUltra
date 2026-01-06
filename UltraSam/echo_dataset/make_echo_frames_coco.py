#!/usr/bin/env python3
import os
import json
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Please install pillow: pip install pillow")

def list_images(frames_dir: Path):
    exts = {".png", ".jpg", ".jpeg"}
    imgs = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in exts])
    if not imgs:
        raise SystemExit(f"No images found in: {frames_dir}")
    return imgs

def rect_segmentation(x, y, w, h):
    # COCO polygon: [x0,y0, x1,y1, x2,y2, x3,y3]
    return [[
        x,     y,
        x + w, y,
        x + w, y + h,
        x,     y + h
    ]]

def build_coco(frames_dir: Path, out_json: Path, file_name_prefix: str = ""):
    images = []
    annotations = []

    category_id = 1
    categories = [{"id": category_id, "name": "object"}]

    imgs = list_images(frames_dir)

    ann_id = 1
    for img_id, img_path in enumerate(imgs, start=1):
        with Image.open(img_path) as im:
            w, h = im.size

        # IMPORTANT:
        # If your mmengine config uses data_prefix.img="" (empty),
        # then file_name should be relative from data_root.
        # Example: "echo_images/echo_frames/xxx.png"
        # If you want just basename, set file_name_prefix="" and also set data_prefix.img="echo_images/echo_frames"
        file_name = f"{file_name_prefix}{img_path.name}"

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": w,
            "height": h
        })

        # Dummy GT: one box covering the FULL image (change if you have real boxes)
        x, y, bw, bh = 0, 0, w, h

        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": category_id,
            "bbox": [x, y, bw, bh],
            "area": float(bw * bh),
            "iscrowd": 0,
            "segmentation": rect_segmentation(x, y, bw, bh)
        })
        ann_id += 1

    coco = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco, f)

    print(f"✅ Wrote {out_json}")
    print(f"Images: {len(images)} | Annotations: {len(annotations)}")
    print(f"Example file_name: {images[0]['file_name']}")

if __name__ == "__main__":
    # === EDIT THESE PATHS IF NEEDED ===
    data_root = Path("/Users/ryancw/Desktop/PlusUltra/UltraSam/echo_dataset")

    # Where your extracted frames actually live:
    frames_dir = data_root / "echo_images" / "echo_frames"

    # Output annotation file:
    out_json = data_root / "echo_frames_coco.json"

    # Because you are using: test_dataloader.dataset.data_prefix.img=""
    # file_name should include the subfolders from data_root:
    # "echo_images/echo_frames/<filename>"
    prefix = "echo_images/echo_frames/"

    build_coco(frames_dir=frames_dir, out_json=out_json, file_name_prefix=prefix)
