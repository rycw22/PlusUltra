# PlusUltra: UltraSAM + EchoNet-Dynamic

This project combines **UltraSAM** and **EchoNet-Dynamic** to enable **accurate heart function analysis from low-quality echocardiograms**.

UltraSAM performs high-quality **left-ventricle segmentation**, and EchoNet-Dynamic then uses those clean segmentations to produce **beat-level cardiac measurements** and **ejection fraction (EF)**.

---

## How to Use it 
(might have slight inconsistencies - check later)
Pre: Download the UltraSAM .pth file from https://github.com/CAMMA-public/UltraSam, and put it into the UltraSAM folder
1. _Put your AVI in a folder, and name it a4c-video-dir_
2. _Run UltraSAM by running the following lines in the terminal:_

  cd UltraSam

  python tools/test.py \
    configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
    UltraSam.pth \
    --show-dir work_dir/my_run

3. _Convert LV frames to AVI by running this:_

  ID="scan01"
  RUN="UltraSam/work_dir/my_run/avi_out/lvonly_frames"

  mkdir -p dynamic/Videos

  ffmpeg -y -framerate 30 -pattern_type glob \
    -i "$RUN/${ID}_*.png" \
    -vf "format=yuvj420p" \
    -c:v mjpeg -q:v 3 \
    "dynamic/Videos/${ID}.avi"

4. _Create FileList.csv by running this:_

  cd dynamic

  cat > FileList.csv << EOF
  FileName,Split,EF
  scan01.avi,TRAIN,0
  scan01.avi,VAL,0
  scan01.avi,TEST,0
  EOF

5. _Convert UltraSAM masks to EchoNet format_

  python scripts/masks_to_volume_tracings.py \
    UltraSam/work_dir/my_run/avi_out/masks \
    scan01.avi


This produces:

VolumeTracings.csv

6. _Run EchoNet_
echonet segmentation \
  --data_dir . \
  --output output_ultrasam \
  --model_name deeplabv3_mobilenet_v3_large \
  --weights ./output/segmentation/deeplabv3_mobilenet_v3_large_random/best.pt \
  --run_test \
  --save_video \
  --num_epochs 0 \
  --batch_size 1 \
  --num_workers 0


_Results appear in:_

dynamic/output_ultrasam/
