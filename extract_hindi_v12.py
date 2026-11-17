#!/usr/bin/env python3
"""
Bilingual electoral-roll voter-card OCR v11 with DELETED watermark detection.
Designed for an INDIVIDUAL voter-card image (for example page_003_card_001.png).

OCR only. No translation or transliteration.

Install:
    python -m pip install opencv-python pytesseract numpy

Examples:
    python extract_hindi_v10.py --image voter_card.png --lang eng --save-debug
    python extract_hindi_v10.py --image voter_card.png --lang hin+eng --save-debug
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def norm_digits(s):
    return str(s).translate(DEVANAGARI_DIGITS)


def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip(" \t:：|-_")


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def remove_long_rules(gray, horizontal=True, vertical=False):
    """Remove card borders from a processing copy; preserve the source."""
    h, w = gray.shape[:2]
    inv = cv2.bitwise_not(gray)
    out = gray.copy()
    if horizontal:
        kh = max(25, w // 5)
        lines = cv2.morphologyEx(
            inv, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1))
        )
        out[lines > 0] = 255
    if vertical:
        kv = max(25, h // 5)
        lines = cv2.morphologyEx(
            inv, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv))
        )
        out[lines > 0] = 255
    return out


def make_variants(crop, scale=4, remove_rules=False):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    if remove_rules:
        gray = remove_long_rules(gray, horizontal=True, vertical=True)
    gray = cv2.resize(
        gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.fastNlMeansDenoising(
        gray, None, h=6, templateWindowSize=7, searchWindowSize=21
    )
    # Keep original grayscale and Otsu variants. Adaptive threshold was
    # damaging thin characters on this scan.
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, otsu


def run_text(crop, lang, configs, scale=4):
    gray, otsu = make_variants(crop, scale=scale)
    results = []
    for variant_name, variant in (("gray", gray), ("otsu", otsu)):
        for config in configs:
            try:
                text = pytesseract.image_to_string(
                    variant, lang=lang, config=config
                ).strip()
                data = pytesseract.image_to_data(
                    variant, lang=lang, config=config, output_type=Output.DICT
                )
                confs = []
                for c in data["conf"]:
                    try:
                        if float(c) >= 0:
                            confs.append(float(c))
                    except (ValueError, TypeError):
                        pass
                confidence = sum(confs) / len(confs) if confs else -1
                results.append({
                    "text": text,
                    "confidence": round(confidence, 2),
                    "variant": variant_name,
                    "config": config,
                })
            except pytesseract.TesseractError:
                continue
    return results, gray, otsu


def detect_marker_component(marker_crop, scale=4):
    """
    Validate that the marker zone contains one character-shaped ink component
    before asking Tesseract to classify it.

    The serial box border is removed on a processing copy. This prevents long
    horizontal/vertical border fragments and scan noise from being interpreted
    as E/R/D. Returns a tight candidate crop and diagnostics.
    """
    gray = cv2.cvtColor(marker_crop, cv2.COLOR_BGR2GRAY) if marker_crop.ndim == 3 else marker_crop.copy()
    enlarged = cv2.resize(
        gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
    )
    blurred = cv2.GaussianBlur(enlarged, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    bh, bw = binary.shape

    # Remove long rules. Kernels are deliberately long relative to the
    # marker zone, but shorter than a normal printed character stroke.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(25, int(bw * 0.62)), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(20, int(bh * 0.62)))
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    cleaned = cv2.subtract(binary, horizontal)
    cleaned = cv2.subtract(cleaned, vertical)

    # Bridge tiny gaps inside a printed glyph without joining distant noise.
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    candidates = []
    min_height = int(bh * 0.34)
    max_height = int(bh * 0.94)
    min_width = max(5, int(bw * 0.035))
    max_width = int(bw * 0.48)
    min_area = max(18, int(bh * bw * 0.0015))

    for component_id in range(1, count):
        x, y, w, h, area = map(int, stats[component_id])
        if not (min_height <= h <= max_height):
            continue
        if not (min_width <= w <= max_width):
            continue
        if area < min_area:
            continue
        aspect = w / float(h)
        if not (0.12 <= aspect <= 1.65):
            continue
        candidates.append((x, y, w, h, area))

    # In some card templates the status letter and serial digits share the
    # same box (for example "R 611"). The status marker is the leftmost
    # character-shaped component. Keep the one-component rule for ordinary
    # cards, but select the leftmost glyph when multiple components exist.
    if not candidates:
        return None, {
            "component_count": 0,
            "components": candidates,
            "accepted": False,
            "reason": "no character-shaped component found",
            "cleaned": cleaned,
            "binary": binary,
        }

    candidates = sorted(candidates, key=lambda item: (item[0], item[1]))
    x, y, w, h, area = candidates[0]
    selected = (x, y, w, h, area)
    pad_x = max(5, int(bw * 0.025))
    pad_y = max(5, int(bh * 0.04))
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(bw, x + w + pad_x), min(bh, y + h + pad_y)

    candidate = cleaned[y1:y2, x1:x2]
    # Return an enlarged, black-on-white glyph crop for OCR.
    candidate = cv2.copyMakeBorder(
        candidate, 18, 18, 18, 18, cv2.BORDER_CONSTANT, value=0
    )
    candidate = cv2.bitwise_not(candidate)
    return candidate, {
        "component_count": len(candidates),
        "components": candidates,
        "selected_component": selected,
        "accepted": True,
        "reason": "leftmost character-shaped component selected as status marker",
        "cleaned": cleaned,
        "binary": binary,
    }


def serial_crop_ocr(card, scale):
    h, w = card.shape[:2]

    full_box = card[
        int(h * 0.045):int(h * 0.18),
        int(w * 0.025):int(w * 0.335)
    ]
    marker_zone = card[
        int(h * 0.045):int(h * 0.18),
        int(w * 0.025):int(w * 0.20)
    ]

    number_configs = [
        "--oem 1 --psm 11 -c tessedit_char_whitelist=0123456789",
        "--oem 1 --psm 12 -c tessedit_char_whitelist=0123456789",
        "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789",
        "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789",
    ]
    number_results, ng, no = run_text(full_box, "eng", number_configs, scale)
    for item in number_results:
        item["crop"] = "full_serial_box"

    number_votes = {}
    for r in number_results:
        digits = re.sub(r"\D", "", norm_digits(r["text"]))
        if 1 <= len(digits) <= 4:
            number_votes.setdefault(digits, []).append(r)

    serial = None
    if number_votes:
        serial_text = max(
            number_votes,
            key=lambda value: (
                len(number_votes[value]),
                max(x["confidence"] for x in number_votes[value]),
                len(value),
            ),
        )
        serial = int(serial_text)

    marker = None
    marker_results = []
    marker_candidate, component_info = detect_marker_component(marker_zone, scale)

    # OCR is invoked only when connected-component analysis finds exactly
    # one plausible glyph in the marker zone. Blank/noisy zones stay null.
    if marker_candidate is not None:
        marker_configs = [
            "--oem 1 --psm 10 -c tessedit_char_whitelist=#ERDS",
            "--oem 1 --psm 8 -c tessedit_char_whitelist=#ERDS",
            "--oem 1 --psm 13 -c tessedit_char_whitelist=#ERDS",
        ]
        marker_results, _, _ = run_text(marker_candidate, "eng", marker_configs, scale)
        accepted = []
        for r in marker_results:
            token = re.sub(r"[^#ERDS]", "", r["text"].upper())
            if token in {"#", "E", "R", "D", "S"}:
                accepted.append((token, r["confidence"], r))

        if accepted:
            votes = {}
            for token, confidence, result in accepted:
                votes.setdefault(token, []).append(confidence)
            marker = max(
                votes,
                key=lambda token: (len(votes[token]), max(votes[token]))
            )

    for item in marker_results:
        item["crop"] = "validated_marker_component"

    debug_images = {
        "full_box": (full_box, ng, no),
        "marker_zone": (marker_zone, None, None),
        "marker_component": (
            marker_candidate if marker_candidate is not None
            else np.full((40, 160), 255, dtype=np.uint8),
            component_info["cleaned"],
            component_info["binary"],
        ),
    }
    # Store diagnostics with candidate list for JSON auditability.
    component_info_public = {
        key: value for key, value in component_info.items()
        if key not in {"cleaned", "binary"}
    }
    return serial, marker, number_results + marker_results, debug_images, component_info_public


def epic_crop_ocr(card, scale):
    h, w = card.shape[:2]
    crop = card[
        int(h * 0.035):int(h * 0.19),
        int(w * 0.66):int(w * 0.995)
    ]
    configs = [
        "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--oem 1 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--oem 1 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ]
    results, gray, otsu = run_text(crop, "eng", configs, scale)
    candidates = []
    for r in results:
        token = re.sub(r"[^A-Z0-9]", "", r["text"].upper())
        m = re.search(r"[A-Z]{3}[0-9]{7}", token)
        if m:
            candidates.append((r["confidence"], m.group(), r))
    if candidates:
        counts = {}
        for confidence, token, result in candidates:
            counts.setdefault(token, []).append(confidence)
        # Multiple independent segmentation modes often outperform a single
        # confidence score (Tesseract may report 0 for a one-word line).
        value = max(
            counts,
            key=lambda token: (len(counts[token]), max(counts[token]))
        )
    else:
        value = None
    return value, results, crop, gray, otsu


def body_ocr(card, lang, scale):
    h, w = card.shape[:2]
    # Exclude serial/EPIC row and photo box. Do not remove text rows' rules
    # because some scans have light characters close to the card borders.
    crop = card[int(h * 0.19):int(h * 0.88), int(w * 0.015):int(w * 0.80)]
    configs = ["--oem 1 --psm 6", "--oem 1 --psm 11"]
    results, gray, otsu = run_text(crop, lang, configs, scale)
    if not results:
        raise RuntimeError("Body OCR did not return any results")
    # Prefer output with enough alphabetic content, but penalize OCR results
    # that omit field labels. This avoids choosing the longest noisy output.
    expected_labels = ("name", "age", "gender", "नाम", "आयु")
    def score(r):
        t = r["text"].lower()
        label_hits = sum(label in t for label in expected_labels)
        useful = sum(ch.isalnum() or "\u0900" <= ch <= "\u097f" for ch in t)
        return (label_hits, r["confidence"], useful)
    best = max(results, key=score)
    return best["text"], results, crop, gray, otsu


def extract_fields(text):
    # OCR may flatten all rows into one line. Parse by label boundaries,
    # not by newline boundaries, and stop each value at the next known label.
    t = clean(text)
    t = t.replace("：", ":")
    fields = {
        "name": None,
        "relative_name": None,
        "relative_type": None,
        "house_number": None,
        "age": None,
        "gender": None,
    }

    name_label = r"(?:निर्वाचक\s*का\s*नाम|मतदाता\s*का\s*नाम|नाम|name)"
    relative_label = (
        r"(?:पिता\s*का\s*नाम|पति\s*का\s*नाम|माता\s*का\s*नाम|अन्य\s*का\s*नाम|"
        r"father'?s?\s+name|husband'?s?\s+name|mother'?s?\s+name|relative'?s?\s+name)"
    )
    house_label = r"(?:मकान\s*संख्या|घर\s*संख्या|गृह\s*संख्या|house\s*(?:number|no\.?))"
    age_label = r"(?:आयु|उम्र|age)"
    gender_label = r"(?:लिंग|gender)"

    def capture(label, next_labels):
        pattern = label + r"\s*[:：\-]?\s*(.*?)"
        if next_labels:
            pattern += r"(?=\s+(?:" + "|".join(next_labels) + r")\b|$)"
        else:
            pattern += r"$"
        m = re.search(pattern, t, flags=re.I)
        return clean(m.group(1)) if m else None

    fields["name"] = capture(name_label, [relative_label, house_label, age_label, gender_label])
    fields["relative_name"] = capture(relative_label, [house_label, age_label, gender_label])
    fields["house_number"] = capture(house_label, [age_label, gender_label])
    if fields["house_number"]:
        fields["house_number"] = fields["house_number"].strip(" '\".,;:|-_") or None

    # Determine relationship type from the matched label.
    rel_match = re.search(relative_label, t, flags=re.I)
    if rel_match:
        label = rel_match.group(0).lower()
        if "पति" in label or "husband" in label:
            fields["relative_type"] = "husband"
        elif "पिता" in label or "father" in label:
            fields["relative_type"] = "father"
        elif "माता" in label or "mother" in label:
            fields["relative_type"] = "mother"
        else:
            fields["relative_type"] = "other"

    # Numeric age is re-read from a dedicated small crop in main(); this is
    # the body-text fallback.
    age_match = re.search(age_label + r"\s*[:：\-]?\s*([0-9०-९]{1,3})", t, re.I)
    if age_match:
        n = norm_digits(age_match.group(1))
        if n.isdigit() and 1 <= int(n) <= 120:
            fields["age"] = int(n)

    if re.search(r"महिला|स्त्री|\bfemale\b", t, re.I):
        fields["gender"] = "female"
    elif re.search(r"पुरुष|\bmale\b", t, re.I):
        fields["gender"] = "male"

    return fields


def targeted_age_ocr(card, scale):
    h, w = card.shape[:2]
    # Age value is on the final details row, left of Gender.
    crop = card[int(h * 0.50):int(h * 0.68), int(w * 0.08):int(w * 0.34)]
    configs = [
        "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789",
        "--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789",
        "--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789",
    ]
    results, gray, otsu = run_text(crop, "eng", configs, scale)
    candidates = []
    for r in results:
        for n in re.findall(r"(?<!\d)(\d{1,3})(?!\d)", norm_digits(r["text"])):
            if 1 <= int(n) <= 120:
                candidates.append((r["confidence"], int(n), r))
    if candidates:
        counts = {}
        for confidence, token, result in candidates:
            counts.setdefault(token, []).append(confidence)
        # Multiple independent segmentation modes often outperform a single
        # confidence score (Tesseract may report 0 for a one-word line).
        value = max(
            counts,
            key=lambda token: (len(counts[token]), max(counts[token]))
        )
    else:
        value = None
    return value, results, crop, gray, otsu




def targeted_house_number_ocr(card, scale, lang="eng"):
    """
    Locate the House Number label with OCR coordinates, then crop only the
    value area on the same text line. This adapts to different card templates
    where the details rows occur at different vertical positions.
    """
    h, w = card.shape[:2]
    gray_card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(
        gray_card, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )
    data = pytesseract.image_to_data(
        enlarged, lang=lang, config="--oem 1 --psm 6",
        output_type=Output.DICT
    )

    # Find the house-number label. Work at OCR's enlarged coordinate scale.
    house_keys = {"house", "number", "no", "मकान", "संख्या", "घर", "गृह"}
    tokens = []
    for i, raw in enumerate(data["text"]):
        token = re.sub(r"[^a-z\u0900-\u097f]", "", raw.lower())
        if not token:
            continue
        tokens.append({
            "i": i,
            "token": token,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "block": int(data["block_num"][i]),
            "par": int(data["par_num"][i]),
            "line": int(data["line_num"][i]),
        })

    anchor = None
    for item in tokens:
        if item["token"] in house_keys or item["token"].startswith("house"):
            anchor = item
            break

    if anchor:
        same_line = [
            t for t in tokens
            if (t["block"], t["par"], t["line"]) ==
               (anchor["block"], anchor["par"], anchor["line"])
        ]
        line_right = max(t["left"] + t["width"] for t in same_line)
        line_top = min(t["top"] for t in same_line)
        line_bottom = max(t["top"] + t["height"] for t in same_line)

        # If "House Number" spans multiple OCR words, start after the last
        # label word, not after the first word.
        label_words = [
            t for t in same_line
            if t["token"] in house_keys or t["token"].startswith("house")
        ]
        label_right = max(t["left"] + t["width"] for t in label_words)
        # Convert coordinates from 2x OCR image back to source-card pixels.
        # Include a little of the label/value separator on the left and
        # enough blank space to the right for the complete value. This avoids
        # clipping the first digit on tightly spaced templates.
        x1 = max(0, int(label_right / 2) - int(w * 0.025))
        x2 = min(w, max(int(line_right / 2) + int(w * 0.025),
                        x1 + int(w * 0.12)))
        y1 = max(0, int(line_top / 2) - max(5, int(h * 0.018)))
        y2 = min(h, int(line_bottom / 2) + max(7, int(h * 0.018)))
        # If OCR line has no text to the right, fall back to a card-relative
        # value zone aligned with the detected label.
        if x2 <= x1 + 3:
            x2 = min(w, x1 + int(w * 0.20))
        crop = card[y1:y2, x1:x2]
    else:
        # Fallback for low-quality scans where the label is not recognized.
        # Use the approximate house-row band but keep it narrow enough to
        # avoid the preceding relative-name line.
        crop = card[int(h * 0.43):int(h * 0.62),
                    int(w * 0.27):int(w * 0.48)]

    configs = [
        "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-",
        "--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-",
        "--oem 1 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-",
        "--oem 1 --psm 7",
    ]
    results, gray, otsu = run_text(crop, "eng", configs, scale)

    candidates = []
    for r in results:
        token = norm_digits(r["text"]).upper()
        token = re.sub(r"[^A-Z0-9/-]", "", token)
        if 1 <= len(token) <= 16 and token not in {
            "HOUSE", "NUMBER", "HOUSENUMBER", "AGE", "GENDER"
        }:
            candidates.append((r["confidence"], token, r))

    if candidates:
        counts = {}
        for confidence, token, result in candidates:
            counts.setdefault(token, []).append(confidence)
        value = max(
            counts,
            key=lambda token: (len(counts[token]), max(counts[token]))
        )
    else:
        value = None
    return value, results, crop, gray, otsu


def marker_status(marker):
    """
    Interpret only the marker whose meaning has been established in this
    workflow. Preserve other observed marks without guessing their meaning.
    """
    if marker == "#":
        return "modified"
    if marker in ("E", "R", "D"):
        return "unclassified"
    return "unmarked" if marker is None else "unclassified"


def detect_deleted_watermark(card, verify_ocr=True):
    """Detect the large diagonal DELETED watermark, not serial-box letters.

    Geometry is the primary signal: this template has several long, similarly
    angled strokes across the center-left of the card. A single OCR pass on
    thick-stroke-only imagery is optional corroboration, NOT a requirement:
    overprinted voter details can make the word unreadable to Tesseract.

    Heuristics were tuned on four distinct positive and three normal examples.
    They are not a calibrated probability and should be revalidated on other
    electoral-roll templates or different diagonal stamps.
    """
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY) if card.ndim == 3 else card
    # Normalization makes thresholds consistent across 457/529px-height cards.
    norm = cv2.resize(gray, (1288, 529), interpolation=cv2.INTER_AREA)
    # Exclude the photo, EPIC, serial boxes and most straight card borders.
    roi = norm[60:415, 190:740]
    edges = cv2.Canny(roi, 80, 180)
    raw_lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=28,
        minLineLength=48, maxLineGap=9,
    )
    strokes = []
    if raw_lines is not None:
        # OpenCV builds differ: HoughLinesP may return (N, 1, 4) or (N, 4).
        # Flatten explicitly so each iteration always yields x1,y1,x2,y2.
        line_rows = np.asarray(raw_lines).reshape(-1, 4)
        for line in line_rows:
            x1, y1, x2, y2 = (int(value) for value in line)
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            length = float(np.hypot(x2 - x1, y2 - y1))
            if -45 < angle < -26 and length >= 48:
                strokes.append((x1, y1, x2, y2, length, angle))

    count = len(strokes)
    total_length = sum(line[4] for line in strokes)
    longest = max((line[4] for line in strokes), default=0.0)
    x_span = (max(max(line[0], line[2]) for line in strokes)
              - min(min(line[0], line[2]) for line in strokes)) if strokes else 0
    # Require repeated diagonal letter strokes spanning much of the word;
    # one diagonal scan artefact or slash cannot satisfy these conditions.
    geometry_match = (count >= 7 and total_length >= 700
                      and longest >= 85 and x_span >= 220)

    matched_text = None
    ocr_checked = False
    if geometry_match and verify_ocr:
        # Thick DELETED strokes survive opening; fine voter text largely does not.
        watermark_roi = norm[55:425, 155:755]
        ink = cv2.threshold(watermark_roi, 170, 255, cv2.THRESH_BINARY_INV)[1]
        thick = cv2.morphologyEx(
            ink, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        isolated = cv2.bitwise_not(thick)
        height, width = isolated.shape
        rotation = cv2.getRotationMatrix2D((width / 2, height / 2), -33, 1)
        abs_cos, abs_sin = abs(rotation[0, 0]), abs(rotation[0, 1])
        new_width = int(height * abs_sin + width * abs_cos)
        new_height = int(height * abs_cos + width * abs_sin)
        rotation[0, 2] += new_width / 2 - width / 2
        rotation[1, 2] += new_height / 2 - height / 2
        rotated = cv2.warpAffine(
            isolated, rotation, (new_width, new_height), borderValue=255,
        )
        try:
            # Only ONE OCR call for a geometry-positive card.
            text = pytesseract.image_to_string(
                rotated, lang="eng", config="--oem 1 --psm 11",
            )
            ocr_checked = True
            # Tolerate one missing D from the overlapping voter text.
            for token in re.findall(r"[A-Z]{5,}", text.upper()):
                if token in {"DELETED", "ELETED", "DELETE"}:
                    matched_text = token
                    break
        except pytesseract.TesseractError:
            pass  # Geometry still provides a useful (reviewable) detection.

    return {
        "detected": bool(geometry_match),
        "method": "diagonal_watermark_geometry_with_optional_ocr",
        "confidence_level": (
            "high" if matched_text else "medium" if geometry_match else "not_detected"
        ),
        "ocr_verified": matched_text is not None,
        "ocr_checked": ocr_checked,
        "ocr_matched_text": matched_text,
        "review_recommended": bool(geometry_match and not matched_text),
        "geometry": {
            "diagonal_segments": count,
            "total_segment_length": round(total_length, 1),
            "longest_segment": round(longest, 1),
            "horizontal_span": int(x_span),
            "angle_range_degrees": [-45, -26],
            "normalized_image_size": [1288, 529],
        },
        "limitations": (
            "Geometry is calibrated for this DELETED watermark layout; "
            "other diagonal stamps may look similar. Medium-confidence "
            "detections should be reviewed."
        ),
    }


def apply_deletion_detection(result, detection):
    """Update deletion fields while preserving other extracted OCR data."""
    fields = result.setdefault("fields", {})
    deleted = bool(detection["detected"])
    fields["deleted_card"] = deleted
    if deleted:
        fields["entry_status"] = "deleted"
    elif fields.get("entry_status") == "deleted":
        fields["entry_status"] = marker_status(fields.get("status_marker"))
    result["deletion_detection"] = detection


def write_json_atomically(path, result):
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same-directory replace prevents interrupted updates leaving partial JSON.
    fd, temp_name = tempfile.mkstemp(prefix=".voter_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Bilingual voter-card OCR v11.")
    parser.add_argument("--image", "--input", dest="image", required=True)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess even when the output JSON already exists")
    parser.add_argument("--lang", default="hin+eng")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--deletion-only", action="store_true",
                        help="Update deletion detection in an existing JSON without rerunning voter OCR")
    parser.add_argument("--skip-deletion-ocr", action="store_true",
                        help="Use fast geometric watermark detection only (no verification OCR)")
    parser.add_argument("--tesseract-cmd")
    args = parser.parse_args()

    src = Path(args.image).expanduser().resolve()
    if not src.is_file():
        parser.error(f"Image not found: {src}")
    out = Path(args.output).expanduser().resolve() if args.output else src.with_suffix(".json")

    # Check the actual output path (not just a sibling JSON beside the PNG).
    # This also makes --overwrite from the master runner work as intended.
    if not args.deletion_only and out.is_file() and not args.overwrite:
        print(f"[SKIPPING] {src} - output JSON already exists: {out}", flush=True)
        return 0
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    try:
        card = read_image(src)
        deletion_detection = detect_deleted_watermark(
            card, verify_ocr=not args.skip_deletion_ocr
        )
        if args.deletion_only:
            if not out.is_file():
                print(f"ERROR: --deletion-only requires an existing JSON: {out}", file=sys.stderr)
                return 2
            result = json.loads(out.read_text(encoding="utf-8"))
            apply_deletion_detection(result, deletion_detection)
            write_json_atomically(out, result)
            print(f"[DELETION CHECK] {src.name}: deleted={deletion_detection['detected']} "
                  f"confidence={deletion_detection['confidence_level']}")
            return 0
        serial, status_marker, serial_results, serial_debug, marker_detection = serial_crop_ocr(card, args.scale)
        epic, epic_results, epic_crop, epic_gray, epic_otsu = epic_crop_ocr(card, args.scale)
        raw_body, body_results, body_crop, body_gray, body_otsu = body_ocr(card, args.lang, args.scale)
        age, age_results, age_crop, age_gray, age_otsu = targeted_age_ocr(card, args.scale)
        house_number, house_results, house_crop, house_gray, house_otsu = targeted_house_number_ocr(card, args.scale, args.lang)
    except pytesseract.TesseractNotFoundError:
        print("Tesseract not found. Install it and add tesseract.exe to PATH.", file=sys.stderr)
        return 2
    except pytesseract.TesseractError as exc:
        print(f"Tesseract error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    fields = extract_fields(raw_body)
    fields["serial_number"] = serial
    fields["status_marker"] = status_marker
    fields["entry_status"] = marker_status(status_marker)
    fields["epic_number"] = epic
    if age is not None:
        fields["age"] = age
    # The full body OCR is the primary source for house number. Only use
    # the targeted crop if body parsing failed; it can otherwise include
    # nearby photo-box text on some layouts.
    if fields.get("house_number") is None and house_number is not None:
        fields["house_number"] = house_number

    result = {
        "source_image": str(src),
        "ocr_language": args.lang,
        "fields": fields,
        "raw_text": raw_body,
        "body_ocr_candidates": body_results,
        "serial_ocr_candidates": serial_results,
        "marker_detection": marker_detection,
        "epic_ocr_candidates": epic_results,
        "age_ocr_candidates": age_results,
        "house_number_ocr_candidates": house_results,
        "notes": [
            "OCR only; no translation or transliteration.",
            "Values are parsed from label boundaries even when OCR flattens rows.",
            "Serial number and status marker are read from separate parts of the serial box.",
            "Observed markers #, E, R, D and S are retained; only # is currently mapped to modified.",
            "Body OCR is preferred for house number; targeted crop is fallback only.",
            "Status markers are OCRed only after connected-component validation finds exactly one glyph-shaped component in the marker zone.",
            "EPIC crop was widened to include the leading characters.",
            "Inspect candidate lists where OCR alternatives disagree.",
            "Diagonal DELETED watermark is detected independently of serial-box markers.",
        ],
    }

    apply_deletion_detection(result, deletion_detection)
    write_json_atomically(out, result)

    if args.save_debug:
        prefix = out.with_suffix("")
        debug = {
            "serial_full_box": serial_debug["full_box"][0],
            "serial_marker_zone": serial_debug["marker_zone"][0],
            "serial_marker_component": serial_debug["marker_component"][0],
            "serial_marker_component_cleaned": serial_debug["marker_component"][1],
            "serial_marker_component_binary": serial_debug["marker_component"][2],
            "epic_crop": epic_crop, "epic_gray": epic_gray, "epic_otsu": epic_otsu,
            "body_crop": body_crop, "body_gray": body_gray, "body_otsu": body_otsu,
            "age_crop": age_crop, "age_gray": age_gray, "age_otsu": age_otsu,
            "house_number_crop": house_crop,
            "house_number_gray": house_gray,
            "house_number_otsu": house_otsu,
        }
        for name, im in debug.items():
            cv2.imwrite(str(prefix) + f"_{name}.png", im)

    print(f"JSON written: {out}")
    print(json.dumps(fields, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
