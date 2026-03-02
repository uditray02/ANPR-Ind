# detect.py
"""
YOLOv5 + PaddleOCR License Plate Recognition
High accuracy pipeline with preprocessing and duplicate filtering.
"""

import argparse
import datetime
import os
import platform
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (
    LOGGER,
    check_img_size,
    increment_path,
    non_max_suppression,
    scale_boxes,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "best.pt",
    source="0",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    device="",
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
):

    source = str(source)
    webcam = source.isnumeric()

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    plates_dir = save_dir / "plates"
    plates_dir.mkdir(parents=True, exist_ok=True)

    plates_txt_path = save_dir / "plates.txt"

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    model.warmup(imgsz=(1, 3, *imgsz))

    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=torch.cuda.is_available(),
        show_log=False,
    )

    total_plates = 0
    detected_plates = set()

    for path, im, im0s, vid_cap, s in dataset:

        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):

            if webcam:
                p, im0 = path[i], im0s[i].copy()
                frame = dataset.count
            else:
                p, im0 = path, im0s.copy()
                frame = getattr(dataset, "frame", 0)

            annotator = Annotator(im0)

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:

                    x1, y1, x2, y2 = map(int, xyxy)

                    # Add padding
                    pad = 10
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(im0.shape[1], x2 + pad)
                    y2 = min(im0.shape[0], y2 + pad)

                    plate_img = im0[y1:y2, x1:x2]

                    if plate_img.size == 0:
                        continue

                    total_plates += 1

                    # Resize for OCR clarity
                    plate_img = cv2.resize(
                        plate_img,
                        None,
                        fx=3,
                        fy=3,
                        interpolation=cv2.INTER_CUBIC,
                    )

                    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

                    # CLAHE contrast boost (better than equalizeHist)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)

                    # Mild denoising
                    gray = cv2.bilateralFilter(gray, 9, 20, 20)

                    result = ocr.ocr(gray, cls=True)

                    plate_text = ""

                    if result and result[0]:
                        texts = []
                        for line in result[0]:
                            text = line[1][0]
                            confidence = line[1][1]
                            if confidence > 0.5:
                                texts.append(text)

                        plate_text = "".join(texts)

                    plate_text = re.sub(r"[^A-Z0-9]", "", plate_text.upper())

                    if len(plate_text) < 5:
                        continue

                    if plate_text not in detected_plates:
                        detected_plates.add(plate_text)

                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        with open(plates_txt_path, "a") as f:
                            f.write(
                                f"{now} | Frame: {frame} | Plate: {plate_text}\n"
                            )

                        crop_name = (
                            f"{Path(p).stem}_{frame}_{plate_text}.jpg"
                        )
                        cv2.imwrite(
                            str(plates_dir / crop_name),
                            plate_img,
                        )

                    label = f"{plate_text} {conf:.2f}"
                    annotator.box_label(
                        xyxy, label, color=colors(int(cls), True)
                    )

            im0 = annotator.result()

            if webcam:
                cv2.imshow("License Plate Detection", im0)
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                save_path = save_dir / Path(p).name
                cv2.imwrite(str(save_path), im0)

    LOGGER.info(f"Total Plates Detected: {total_plates}")
    LOGGER.info(f"Results saved to {save_dir}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "best.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default=ROOT / "runs/detect")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
