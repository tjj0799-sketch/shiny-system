#!/usr/bin/env python3
"""
批量人像 AI 分析 + 美颜 + 调色（分步骤执行版）

步骤：
1) 读取输入文件夹图片
2) AI 分析人物（人脸检测 + 人脸数量统计）
3) 自动美颜（双边滤波 + 轻度磨皮 + 提亮）
4) 自动调色（白平衡 + 对比度 + 饱和度）
5) 输出到新文件夹
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class FaceAnalysis:
    faces: int
    boxes: List[Tuple[int, int, int, int]]


def list_images(input_dir: Path) -> List[Path]:
    images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()]
    return images


def load_face_detector() -> cv2.CascadeClassifier:
    model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(model_path)
    if detector.empty():
        raise RuntimeError(f"无法加载人脸检测模型: {model_path}")
    return detector


def analyze_faces(image_bgr: np.ndarray, detector: cv2.CascadeClassifier) -> FaceAnalysis:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    result = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]
    return FaceAnalysis(faces=len(result), boxes=result)


def skin_smooth(image_bgr: np.ndarray, strength: float = 0.55) -> np.ndarray:
    # 双边滤波: 保边磨皮
    d = 9
    sigma_color = 75
    sigma_space = 75
    smooth = cv2.bilateralFilter(image_bgr, d, sigma_color, sigma_space)
    out = cv2.addWeighted(smooth, strength, image_bgr, 1 - strength, 0)

    # 轻微提亮
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def gray_world_white_balance(image_bgr: np.ndarray) -> np.ndarray:
    img = image_bgr.astype(np.float32)
    b_avg, g_avg, r_avg = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
    gray = (b_avg + g_avg + r_avg) / 3.0
    kb = gray / (b_avg + 1e-6)
    kg = gray / (g_avg + 1e-6)
    kr = gray / (r_avg + 1e-6)
    img[:, :, 0] *= kb
    img[:, :, 1] *= kg
    img[:, :, 2] *= kr
    return np.clip(img, 0, 255).astype(np.uint8)


def auto_color_grade(image_bgr: np.ndarray) -> np.ndarray:
    # 白平衡
    wb = gray_world_white_balance(image_bgr)

    # 对比度增强（LAB 的 L 通道 CLAHE）
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    contrast = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # 饱和度轻微增强
    hsv = cv2.cvtColor(contrast, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.10, 0, 255)
    graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return graded


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(input_dir: Path, output_dir: Path, analysis_file: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = load_face_detector()

    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"输入目录没有图片: {input_dir}")

    report = {"input_dir": str(input_dir), "output_dir": str(output_dir), "files": []}

    print("[步骤1] 已读取图片数量:", len(images))

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[跳过] 无法读取: {img_path.name}")
            continue

        # 步骤2: AI 分析人物
        face_info = analyze_faces(img, detector)

        # 步骤3: 自动美颜
        beauty = skin_smooth(img)

        # 步骤4: 自动调色
        final_img = auto_color_grade(beauty)

        # 步骤5: 输出
        out_path = output_dir / img_path.name
        ok = cv2.imwrite(str(out_path), final_img)
        if not ok:
            print(f"[失败] 写入失败: {out_path}")
            continue

        report["files"].append(
            {
                "file": img_path.name,
                "faces": face_info.faces,
                "face_boxes": face_info.boxes,
                "output": out_path.name,
            }
        )
        print(f"[{idx}/{len(images)}] 完成: {img_path.name} -> {out_path.name} (faces={face_info.faces})")

    save_json(analysis_file, report)
    print(f"\n处理完成。分析报告已保存: {analysis_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量人像 AI 分析 + 美颜 + 调色")
    parser.add_argument("--input", required=True, help="输入图片文件夹")
    parser.add_argument("--output", required=True, help="输出图片文件夹")
    parser.add_argument("--report", default="analysis_report.json", help="分析报告 JSON 文件路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(Path(args.input), Path(args.output), Path(args.report))
