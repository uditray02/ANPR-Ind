# detect_paddle_improved.py
"""
YOLOv5 + PaddleOCR License Plate Recognition
Enhanced accuracy pipeline with advanced preprocessing and ensemble OCR.

DEFAULT RESOLUTION: 1280x1280 (2x standard for better distant/small plate detection)
- Use --imgsz 640 for faster processing on CPU
- Use --imgsz 1920 for maximum detection quality
- Use --imgsz 2560 for very distant plates (slow but thorough)
"""

import argparse
import datetime
import os
import platform
import re
import sys
from collections import Counter
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


def deskew_plate(plate_img):
    """
    Correct plate angle/rotation using Hough Line Transform.
    
    Args:
        plate_img: Input plate image
        
    Returns:
        Deskewed plate image
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)
    
    if lines is not None and len(lines) > 0:
        # Calculate median angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)
        
        median_angle = np.median(angles)
        
        # Only rotate if angle is significant (> 2 degrees)
        if abs(median_angle) > 2:
            # Rotate image
            (h, w) = plate_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                plate_img,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return rotated
    
    return plate_img


def preprocess_plate(plate_img, fast_mode=False):
    """
    Apply multiple preprocessing techniques for better OCR accuracy.
    
    Args:
        plate_img: Input plate image
        fast_mode: If True, use only 2 best methods for speed
        
    Returns:
        List of preprocessed images using different methods
    """
    # Resize first for better detail
    plate_img = cv2.resize(
        plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
    )
    
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    preprocessed = []
    
    # Method 1: CLAHE + Bilateral Filter (Best for most cases)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced1 = clahe.apply(gray)
    enhanced1 = cv2.bilateralFilter(enhanced1, 9, 75, 75)
    preprocessed.append(enhanced1)
    
    if fast_mode:
        # In fast mode, only use one more method
        # Method 2: Adaptive Thresholding (Fast and effective)
        enhanced2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        preprocessed.append(enhanced2)
        return preprocessed
    
    # Full mode: Use all methods
    # Method 2: Adaptive Thresholding (Gaussian)
    enhanced2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    preprocessed.append(enhanced2)
    
    # Method 3: Otsu's Binarization
    _, enhanced3 = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    preprocessed.append(enhanced3)
    
    # Method 4: Morphological operations + Median Blur
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    enhanced4 = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    enhanced4 = cv2.medianBlur(enhanced4, 3)
    # Apply CLAHE to morphological result
    enhanced4 = clahe.apply(enhanced4)
    preprocessed.append(enhanced4)
    
    # Method 5: Contrast stretching
    enhanced5 = cv2.normalize(
        gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    enhanced5 = cv2.GaussianBlur(enhanced5, (3, 3), 0)
    preprocessed.append(enhanced5)
    
    return preprocessed


def clean_plate_text(text):
    """
    Clean and fix common OCR errors in license plates.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned plate text
    """
    # Remove non-alphanumeric characters
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    
    # Fix common OCR mistakes based on context
    cleaned = []
    for i, char in enumerate(text):
        # Check if surrounded by digits
        prev_is_digit = i > 0 and text[i - 1].isdigit()
        next_is_digit = i < len(text) - 1 and text[i + 1].isdigit()
        
        # Context-aware character replacement
        if char == "O" and (prev_is_digit or next_is_digit):
            cleaned.append("0")
        elif char == "I" and (prev_is_digit or next_is_digit):
            cleaned.append("1")
        elif char == "Z" and (prev_is_digit or next_is_digit):
            cleaned.append("2")
        elif char == "S" and (prev_is_digit or next_is_digit):
            cleaned.append("5")
        elif char == "B" and (prev_is_digit or next_is_digit):
            cleaned.append("8")
        elif char == "G" and (prev_is_digit or next_is_digit):
            cleaned.append("6")
        else:
            cleaned.append(char)
    
    return "".join(cleaned)


def ocr_with_voting(plate_img, ocr, confidence_threshold=0.6, fast_mode=False):
    """
    Run OCR on multiple preprocessed versions and vote on the best result.
    
    Args:
        plate_img: Input plate image
        ocr: PaddleOCR instance
        confidence_threshold: Minimum confidence for accepting OCR results
        fast_mode: If True, use fewer preprocessing methods and early stopping
        
    Returns:
        Tuple of (plate_text, confidence)
    """
    preprocessed_images = preprocess_plate(plate_img, fast_mode=fast_mode)
    
    results = []
    
    for img in preprocessed_images:
        try:
            result = ocr.ocr(img, cls=True)
            
            if result and result[0]:
                texts = []
                confidences = []
                
                for line in result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    if confidence > confidence_threshold:
                        texts.append(text)
                        confidences.append(confidence)
                
                if texts:
                    plate_text = "".join(texts)
                    plate_text = clean_plate_text(plate_text)
                    
                    # Only add if text is reasonable length
                    if 5 <= len(plate_text) <= 10:
                        avg_conf = sum(confidences) / len(confidences)
                        results.append((plate_text, avg_conf))
                        
                        # Early stopping in fast mode if confidence is very high
                        if fast_mode and avg_conf > 0.85:
                            return plate_text, avg_conf
        
        except Exception as e:
            # Skip this preprocessing method if OCR fails
            continue
    
    # Vote: return most common result with highest confidence
    if not results:
        return "", 0.0
    
    # Count occurrences of each plate text
    plate_texts = [r[0] for r in results]
    text_counts = Counter(plate_texts)
    
    # Get the most common text
    most_common_text = text_counts.most_common(1)[0][0]
    
    # Get the highest confidence for the most common result
    best_conf = max(r[1] for r in results if r[0] == most_common_text)
    
    return most_common_text, best_conf


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
    ocr_conf_thres=0.7,
    min_plate_length=5,
    max_plate_length=10,
    save_crops=True,
    fast_mode=False,
):
    """
    Run license plate detection and OCR.
    
    Args:
        weights: Path to YOLOv5 weights
        source: Input source (webcam index, video file, or image directory)
        imgsz: Inference image size
        conf_thres: Object detection confidence threshold
        iou_thres: IOU threshold for NMS
        device: CUDA device or 'cpu'
        project: Save directory project path
        name: Save directory name
        exist_ok: Overwrite existing directory
        ocr_conf_thres: OCR confidence threshold
        min_plate_length: Minimum plate text length
        max_plate_length: Maximum plate text length
        save_crops: Save cropped plate images
        fast_mode: Enable CPU-friendly mode (2 preprocessing methods instead of 5)
    """
    source = str(source)
    webcam = source.isnumeric()

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    plates_dir = save_dir / "plates"
    if save_crops:
        plates_dir.mkdir(parents=True, exist_ok=True)

    plates_txt_path = save_dir / "plates.txt"

    # Initialize device and model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Initialize dataset
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Warmup
    model.warmup(imgsz=(1, 3, *imgsz))

    # Initialize PaddleOCR with CPU-optimized parameters
    use_gpu = torch.cuda.is_available() and device != 'cpu'
    
    if fast_mode or not use_gpu:
        # CPU-friendly configuration
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            rec_batch_num=1,  # Reduced for CPU
            max_text_length=15,  # Reduced for speed
            use_space_char=False,
            enable_mkldnn=True,  # Intel CPU optimization
            cpu_threads=4,  # Use multiple CPU threads
        )
        LOGGER.info("PaddleOCR initialized in CPU-optimized mode")
    else:
        # GPU configuration
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=True,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            rec_batch_num=6,
            max_text_length=25,
            use_space_char=False,
        )
        LOGGER.info("PaddleOCR initialized in GPU mode")

    total_plates = 0
    detected_plates = set()

    mode_str = "FAST (CPU-friendly)" if fast_mode else "ACCURATE"
    LOGGER.info(f"Starting detection in {mode_str} mode...")

    for path, im, im0s, vid_cap, s in dataset:

        # Prepare image
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im[None]

        # Inference
        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Process detections
        for i, det in enumerate(pred):

            if webcam:
                p, im0 = path[i], im0s[i].copy()
                frame = dataset.count
            else:
                p, im0 = path, im0s.copy()
                frame = getattr(dataset, "frame", 0)

            annotator = Annotator(im0)

            if len(det):
                # Rescale boxes
                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], im0.shape
                ).round()

                for *xyxy, conf, cls in det:

                    x1, y1, x2, y2 = map(int, xyxy)

                    # Add adaptive padding based on box size
                    box_width = x2 - x1
                    box_height = y2 - y1
                    pad_x = max(5, int(box_width * 0.05))
                    pad_y = max(5, int(box_height * 0.05))
                    
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(im0.shape[1], x2 + pad_x)
                    y2 = min(im0.shape[0], y2 + pad_y)

                    plate_img = im0[y1:y2, x1:x2]

                    if plate_img.size == 0:
                        continue

                    total_plates += 1

                    # Step 1: Deskew the plate (skip in fast mode for speed)
                    if not fast_mode:
                        plate_img = deskew_plate(plate_img)

                    # Step 2: Run OCR with voting ensemble
                    plate_text, ocr_confidence = ocr_with_voting(
                        plate_img, ocr, confidence_threshold=0.6, fast_mode=fast_mode
                    )

                    # Validation checks
                    if not plate_text:
                        continue
                    
                    if len(plate_text) < min_plate_length or len(plate_text) > max_plate_length:
                        continue
                    
                    if ocr_confidence < ocr_conf_thres:
                        continue

                    # Check for duplicates
                    if plate_text not in detected_plates:
                        detected_plates.add(plate_text)

                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Save to text file
                        with open(plates_txt_path, "a") as f:
                            f.write(
                                f"{now} | Frame: {frame} | Plate: {plate_text} | Confidence: {ocr_confidence:.3f}\n"
                            )

                        # Save cropped image
                        if save_crops:
                            crop_name = f"{Path(p).stem}_{frame}_{plate_text}.jpg"
                            # Save the preprocessed version for reference
                            preprocessed = preprocess_plate(plate_img, fast_mode=fast_mode)
                            cv2.imwrite(
                                str(plates_dir / crop_name),
                                preprocessed[0],  # Save best preprocessed version
                            )

                        LOGGER.info(
                            f"Detected: {plate_text} (Conf: {ocr_confidence:.3f})"
                        )

                    # Annotate image
                    label = f"{plate_text} {ocr_confidence:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

            # Display or save result
            im0 = annotator.result()

            if webcam:
                cv2.imshow("License Plate Detection", im0)
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                save_path = save_dir / Path(p).name
                cv2.imwrite(str(save_path), im0)

    # Cleanup
    if webcam:
        cv2.destroyAllWindows()

    # Summary
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Detection Complete!")
    LOGGER.info(f"Mode: {mode_str}")
    LOGGER.info(f"Total Plates Detected: {total_plates}")
    LOGGER.info(f"Unique Plates Recognized: {len(detected_plates)}")
    LOGGER.info(f"Results saved to: {save_dir}")
    LOGGER.info(f"{'='*60}\n")


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default=ROOT / "best.pt", help="Model weights path"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Source: 0 for webcam, video file, or image directory",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[1280],  # Increased from 640 for better plate detection
        help="Inference size (pixels) - higher = better detection but slower",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Object detection confidence threshold",
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="", help="CUDA device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--project", default=ROOT / "runs/detect", help="Save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="Save results to project/name")
    parser.add_argument(
        "--exist-ok", action="store_true", help="Existing project/name ok"
    )
    parser.add_argument(
        "--ocr-conf-thres",
        type=float,
        default=0.7,
        help="OCR confidence threshold",
    )
    parser.add_argument(
        "--min-plate-length",
        type=int,
        default=5,
        help="Minimum plate text length",
    )
    parser.add_argument(
        "--max-plate-length",
        type=int,
        default=10,
        help="Maximum plate text length",
    )
    parser.add_argument(
        "--no-save-crops",
        action="store_true",
        help="Do not save cropped plate images",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable CPU-friendly mode (2 preprocessing methods, no deskewing, early stopping)",
    )
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    opt.save_crops = not opt.no_save_crops
    
    # Remove no_save_crops since run() doesn't expect it
    delattr(opt, 'no_save_crops')
    
    return opt


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
