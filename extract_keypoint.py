"""
3stream_mydata.py
=================
Trich xuat 3-stream skeleton (joint / motion / bone-angle) tu dataset:
  Dataset/
    fall/
      lying_fall/      *.mp4
      sitting_fall/    *.mp4
      standing_fall/   *.mp4
    non_fall/          *.mp4

Nhan: thu muc fall/ -> label=1, non_fall/ -> label=0

Output duy nhat:
  custom_all.npz - tat ca samples + metadata
                   Keys: X_joint, X_motion, X_angle, y,
                         img_w, img_h, seq_id, subject, is_flip, category

Toa do: x,y pixel tho (khong normalize, khong can chinh centroid).
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# ==========================================================
#                         CONFIG
# ==========================================================
DATASET_ROOT = r"Dataset"   # thu muc goc chua fall/ va non_fall/
OUTPUT_DIR   = r"."

FALL_DIR  = "fall"
NFALL_DIR = "non_fall"

FPS         = 30
SEQ_LEN     = 60
MIN_FRAMES  = 30
MIN_DETECTED_FRAMES = 30
MIN_VALID_FRAME_RATIO = 0.5
FALL_STRIDE = SEQ_LEN       # khÃ´ng overlap
CONF_THRESH = 0.1

# CÃ¡c loáº¡i ngÃ£ Ä‘Æ°á»£c há»— trá»£ (khá»›p vá»›i tÃªn thÆ° má»¥c)
FALL_CATEGORIES = ["Lying_Falls", "Sitting_Falls", "Standing_Falls"]
CATEGORY_NAME_MAP = {
    "lying_fall": "Lying_Falls",
    "sitting_fall": "Sitting_Falls",
    "standing_fall": "Standing_Falls",
}

# ==========================================================
#  KEYPOINT MAPPING  (14 joints: 13 real + 1 virtual Mid-Hip)
# ==========================================================
mp_pose = mp.solutions.pose

KEYPOINT_ORDER = [
    'Nose',           # 0
    'Left Shoulder',  # 1
    'Right Shoulder', # 2
    'Left Elbow',     # 3
    'Right Elbow',    # 4
    'Left Wrist',     # 5
    'Right Wrist',    # 6
    'Left Hip',       # 7
    'Right Hip',      # 8
    'Left Knee',      # 9
    'Right Knee',     # 10
    'Left Ankle',     # 11
    'Right Ankle',    # 12
    # 13: Mid-Hip (virtual) = (Left Hip + Right Hip) / 2
]

LANDMARK_MAP = {
    name: getattr(mp_pose.PoseLandmark, name.upper().replace(" ", "_"))
    for name in KEYPOINT_ORDER[:13]
}

NUM_KEYPOINTS = 14

PARENT_MAP = {
    13: None,
    7: 13,  8: 13,
    1:  7,  2:  8,
    0:  1,
    3:  1,  4:  2,
    5:  3,  6:  4,
    9:  7,  10: 8,
    11: 9,  12: 10,
}

# ==========================================================
#             3-STREAM COMPUTATION
# ==========================================================

def extract_keypoints_single(landmarks, img_w: int, img_h: int):
    """TrÃ­ch 14 keypoints â€” tá»a Ä‘á»™ pixel thÃ´ (x,y raw)."""
    kps = []
    for name in KEYPOINT_ORDER[:13]:
        try:
            lm = landmarks.landmark[LANDMARK_MAP[name].value]
            kps.append((
                lm.x * img_w,
                lm.y * img_h,
                getattr(lm, "visibility", getattr(lm, "presence", 0.0))
            ))
        except Exception:
            return None

    arr = np.array(kps, dtype=np.float32)   # (13, 3)

    mid_xy   = (arr[7, :2] + arr[8, :2]) / 2.0
    mid_conf = np.minimum(arr[7, 2], arr[8, 2])
    mid_hip  = np.array([mid_xy[0], mid_xy[1], mid_conf], dtype=np.float32)

    return np.vstack([arr, mid_hip])   # (14, 3)


def compute_motion(joint_feat):
    """Temporal difference stream. Shape: (T, 14, 3)."""
    motion = np.zeros_like(joint_feat)
    motion[1:] = joint_feat[1:] - joint_feat[:-1]
    motion[:, :, 2] = joint_feat[:, :, 2]
    return motion


def compute_bone_angle(joint_feat):
    """Bone angle stream. Shape: (T, 14, 3)."""
    F, V, _ = joint_feat.shape
    angle = np.zeros((F, V, 3), dtype=np.float32)
    for v in range(V):
        p = PARENT_MAP[v]
        if p is None:
            angle[:, v, 0] = 1.0
            angle[:, v, 2] = joint_feat[:, v, 2]
        else:
            dy = joint_feat[:, v, 1] - joint_feat[:, p, 1]
            dx = joint_feat[:, v, 0] - joint_feat[:, p, 0]
            th = np.arctan2(dy, dx)
            angle[:, v, 0] = np.cos(th)
            angle[:, v, 1] = np.sin(th)
            angle[:, v, 2] = np.minimum(joint_feat[:, v, 2],
                                         joint_feat[:, p, 2])
    return angle


def interpolate_missing_keypoint_frames(raw_kps):
    """
    Fill frames without detected keypoints from neighboring valid frames.
    Missing frames inside the sequence are linearly interpolated; missing frames
    at the beginning/end are repeated from the nearest valid frame.
    Confidence is interpolated too, so the 3-channel skeleton remains complete.
    """
    if not raw_kps:
        return None, None, 0

    F = len(raw_kps)
    arr = np.full((F, NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
    valid_mask = np.zeros(F, dtype=bool)

    for i, kp in enumerate(raw_kps):
        if kp is not None:
            arr[i] = kp
            valid_mask[i] = True

    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return None, None, 0

    frame_idx = np.arange(F)
    valid_idx = frame_idx[valid_mask]
    filled = np.empty_like(arr)
    for v in range(NUM_KEYPOINTS):
        for c in range(3):
            filled[:, v, c] = np.interp(frame_idx, valid_idx, arr[valid_mask, v, c])

    return filled.astype(np.float32), valid_mask, valid_count


# ==========================================================
#        SKELETON EXTRACTION tá»« file VIDEO
# ==========================================================

def extract_valid_skeletons_from_video(video_path: Path):
    """
    TrÃ­ch skeleton tá»« tá»«ng frame video báº±ng MediaPipe.
    Frame khÃ´ng detect Ä‘Æ°á»£c â†’ bá» qua (khÃ´ng interpolate).
    Tráº£ vá»: (skel_array, img_w, img_h) hoáº·c (None, w, h) náº¿u quÃ¡ Ã­t frame.
    skel_array shape: (F_valid, 14, 3), tá»a Ä‘á»™ pixel thÃ´.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, 0, 0

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Keep one slot per original frame; missing detections are interpolated later.
    raw_kps = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=False,        # táº¯t smooth trÃ¡nh data leak
        min_detection_confidence=CONF_THRESH,
        min_tracking_confidence=CONF_THRESH,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                kp = extract_keypoints_single(res.pose_landmarks, img_w, img_h)
                raw_kps.append(kp)
            else:
                raw_kps.append(None)

    cap.release()

    if len(raw_kps) < MIN_FRAMES:
        return None, None, img_w, img_h

    arr, detected_mask, valid_count = interpolate_missing_keypoint_frames(raw_kps)
    if arr is None or valid_count < MIN_DETECTED_FRAMES:
        return None, None, img_w, img_h

    return arr, detected_mask, img_w, img_h


# ==========================================================
#                    WINDOWING
# ==========================================================

def pad_to_seq_len(arr):
    """Repeat the last frame until shape is (SEQ_LEN, 14, 3)."""
    F = arr.shape[0]
    if F >= SEQ_LEN:
        return arr[:SEQ_LEN]
    # No zero padding: extend short clips by repeating the last real frame.
    pad = np.repeat(arr[-1:, ...], SEQ_LEN - F, axis=0)
    return np.concatenate([arr, pad])


def pad_mask_to_seq_len(mask):
    """Repeat the last detection flag until shape is (SEQ_LEN,)."""
    F = mask.shape[0]
    if F >= SEQ_LEN:
        return mask[:SEQ_LEN]
    pad = np.repeat(mask[-1:], SEQ_LEN - F, axis=0)
    return np.concatenate([mask, pad])


def window_valid_ratio(win):
    """
    TÃ­nh tá»‰ lá»‡ frame há»£p lá»‡ trong window.
    Frame há»£p lá»‡ = mean confidence cá»§a táº¥t cáº£ keypoint > CONF_THRESH.
    win shape: (SEQ_LEN, 14, 3)
    """
    return float(win.sum()) / float(win.shape[0])


def make_fall_windows(skel, detected_mask):
    """Cáº¯t khÃ´ng overlap stride=SEQ_LEN. F < SEQ_LEN â†’ 1 window láº·p frame cuá»‘i.
    Lá»c bá» window cÃ³ tá»‰ lá»‡ frame há»£p lá»‡ < MIN_VALID_FRAME_RATIO.
    """
    F = len(skel)
    wins = []
    if F < MIN_FRAMES:
        return wins

    if F < SEQ_LEN:
        w = pad_to_seq_len(skel)
        wm = pad_mask_to_seq_len(detected_mask)
        ratio = window_valid_ratio(wm)
        if ratio >= MIN_VALID_FRAME_RATIO:
            wins.append(w)
        else:
            tqdm.write(f"  [WARN] Drop short window: valid_ratio="
                       f"{ratio:.2f} < {MIN_VALID_FRAME_RATIO}")
        return wins

    p = 0
    while p + SEQ_LEN <= F:
        w = skel[p:p + SEQ_LEN].copy()
        wm = detected_mask[p:p + SEQ_LEN]
        ratio = window_valid_ratio(wm)
        if ratio >= MIN_VALID_FRAME_RATIO:
            wins.append(w)
        else:
            tqdm.write(f"  [WARN] Drop fall window @frame{p}: "
                       f"valid_ratio={ratio:.2f} < {MIN_VALID_FRAME_RATIO}")
        p += FALL_STRIDE

    return wins


def make_nonfall_windows(skel, detected_mask):
    """Cáº¯t liÃªn tiáº¿p khÃ´ng overlap. F < SEQ_LEN â†’ 1 window láº·p frame cuá»‘i.
    Äoáº¡n Ä‘uÃ´i < SEQ_LEN sau cÃ¡c cá»­a sá»• Ä‘á»§ 60 frame â†’ bá».
    Lá»c bá» window cÃ³ tá»‰ lá»‡ frame há»£p lá»‡ < MIN_VALID_FRAME_RATIO.
    """
    F = len(skel)
    wins = []
    if F < MIN_FRAMES:
        return wins

    if F < SEQ_LEN:
        w = pad_to_seq_len(skel)
        wm = pad_mask_to_seq_len(detected_mask)
        ratio = window_valid_ratio(wm)
        if ratio >= MIN_VALID_FRAME_RATIO:
            wins.append(w)
        else:
            tqdm.write(f"  [WARN] Drop short non-fall window: "
                       f"valid_ratio={ratio:.2f} < {MIN_VALID_FRAME_RATIO}")
        return wins

    p = 0
    while p + SEQ_LEN <= F:
        w = skel[p:p + SEQ_LEN].copy()
        wm = detected_mask[p:p + SEQ_LEN]
        ratio = window_valid_ratio(wm)
        if ratio >= MIN_VALID_FRAME_RATIO:
            wins.append(w)
        else:
            tqdm.write(f"  [WARN] Drop non-fall window @frame{p}: "
                       f"valid_ratio={ratio:.2f} < {MIN_VALID_FRAME_RATIO}")
        p += SEQ_LEN

    return wins


# ==========================================================
#   THU THáº¬P VIDEO Tá»ª Cáº¤U TRÃšC THÆ¯ Má»¤C
# ==========================================================

def collect_videos(dataset_root: Path):
    """Collect videos for the current Dataset/ folder structure."""
    videos = []
    default_subject = "all"

    fall_root = dataset_root / FALL_DIR
    if fall_root.exists():
        for category_dir in sorted(fall_root.iterdir()):
            if not category_dir.is_dir():
                continue
            category = CATEGORY_NAME_MAP.get(category_dir.name, category_dir.name)
            for vf in sorted(category_dir.rglob("*.mp4")):
                videos.append({
                    'path':     vf,
                    'label':    1,
                    'subject':  default_subject,
                    'is_flip':  vf.stem.lower().endswith("_flip"),
                    'category': category,
                })

    nfall_root = dataset_root / NFALL_DIR
    if nfall_root.exists():
        for vf in sorted(nfall_root.rglob("*.mp4")):
            videos.append({
                'path':     vf,
                'label':    0,
                'subject':  default_subject,
                'is_flip':  vf.stem.lower().endswith("_flip"),
                'category': 'non_fall',
            })

    return videos


# ==========================================================
#                        MAIN
# ==========================================================

def main():
    print("=" * 70)
    print("  Custom Fall Dataset â€“ 3-Stream Preprocessing")
    print("  [pixel coords thÃ´ | no smooth | no overlap | per-category]")
    print("=" * 70)
    print(f"  DATASET_ROOT : {os.path.abspath(DATASET_ROOT)}")
    print(f"  OUTPUT_DIR   : {os.path.abspath(OUTPUT_DIR)}")
    print(f"  SEQ_LEN      : {SEQ_LEN}f @ {FPS}fps = {SEQ_LEN/FPS:.1f}s")
    print(f"  MIN_FRAMES   : {MIN_FRAMES}")
    print(f"  MIN_DETECTED : {MIN_DETECTED_FRAMES}")
    print(f"  CONF_THRESH  : {CONF_THRESH}")
    print(f"  Categories   : {FALL_CATEGORIES + ['non_fall']}")
    print()

    root = Path(DATASET_ROOT)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not root.exists():
        print(f"[ERROR] DATASET_ROOT khÃ´ng tá»“n táº¡i: {root.resolve()}")
        return


    all_videos = collect_videos(root)
    if not all_videos:
        print("[ERROR] KhÃ´ng tÃ¬m tháº¥y video nÃ o. Kiá»ƒm tra cáº¥u trÃºc thÆ° má»¥c.")
        return

    print(f"Tá»•ng video tÃ¬m tháº¥y: {len(all_videos)}")
    video_cat_counts = {}
    for v in all_videos:
        video_cat_counts[v['category']] = video_cat_counts.get(v['category'], 0) + 1
    for cat, cnt in sorted(video_cat_counts.items()):
        print(f"  {cat}: {cnt} videos")
    print()

    # â”€â”€ TrÃ­ch xuáº¥t skeleton cho Táº¤T Cáº¢ video â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    all_joints, all_motions, all_angles = [], [], []
    all_labels, all_iw, all_ih          = [], [], []
    all_seq_ids                          = []
    all_subjects_list                    = []
    all_is_flip                          = []
    all_categories_list                  = []
    seq_meta  = []
    seq_id    = 0
    err_count = 0

    print(f"{'â”€'*60}")
    print("TrÃ­ch xuáº¥t skeleton tá»« táº¥t cáº£ video...")

    for vinfo in tqdm(all_videos, desc="  Extracting"):
        vpath    = vinfo['path']
        label    = vinfo['label']
        subject  = vinfo['subject']
        is_flip  = vinfo['is_flip']
        category = vinfo['category']

        skel, detected_mask, img_w, img_h = extract_valid_skeletons_from_video(vpath)

        if skel is None:
            tqdm.write(f"  [WARN] QuÃ¡ Ã­t frame detect: {vpath.name}")
            err_count += 1
            seq_id += 1
            continue

        wins = make_fall_windows(skel, detected_mask) if label == 1 else make_nonfall_windows(skel, detected_mask)

        if not wins:
            seq_id += 1
            continue

        for w in wins:
            all_joints.append(w)
            all_motions.append(compute_motion(w))
            all_angles.append(compute_bone_angle(w))
            all_labels.append(label)
            all_iw.append(img_w)
            all_ih.append(img_h)
            all_seq_ids.append(seq_id)
            all_subjects_list.append(subject)
            all_is_flip.append(int(is_flip))
            all_categories_list.append(category)

        seq_meta.append({
            'seq_id':   seq_id,
            'subject':  subject,
            'name':     vpath.stem,
            'label':    label,
            'n_wins':   len(wins),
            'is_flip':  is_flip,
            'category': category,
        })
        seq_id += 1

    if not all_labels:
        print("[ERROR] KhÃ´ng cÃ³ sample nÃ o.")
        return

    # â”€â”€ Build arrays â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_joint      = np.array(all_joints,           dtype=np.float32)
    X_motion     = np.array(all_motions,          dtype=np.float32)
    X_angle      = np.array(all_angles,           dtype=np.float32)
    y            = np.array(all_labels,           dtype=np.int64)
    img_w_arr    = np.array(all_iw,               dtype=np.int32)
    img_h_arr    = np.array(all_ih,               dtype=np.int32)
    seq_ids      = np.array(all_seq_ids,          dtype=np.int32)
    subject_arr  = np.array(all_subjects_list,    dtype='U32')
    is_flip_arr  = np.array(all_is_flip,          dtype=np.int32)
    category_arr = np.array(all_categories_list,  dtype='U32')

    # Sanity check
    xy = X_joint[:, :, :2]
    print(f"\n  [Sanity] x,y range: [{xy.min():.2f}, {xy.max():.2f}]  "
          f"(expected pixel scale)")

    # Per-category sample count
    print(f"\n  [Category distribution]")
    for cat in FALL_CATEGORIES + ['non_fall']:
        n = int((category_arr == cat).sum())
        print(f"    {cat:<22}: {n} samples")

    out_all = os.path.join(OUTPUT_DIR, "custom_all.npz")
    np.savez_compressed(
        out_all,
        X_joint  = X_joint,
        X_motion = X_motion,
        X_angle  = X_angle,
        y        = y,
        img_w    = img_w_arr,
        img_h    = img_h_arr,
        seq_id   = seq_ids,
        subject  = subject_arr,
        is_flip  = is_flip_arr,
        category = category_arr,
    )
    print(f"\n  -> {os.path.basename(out_all)}  shape={X_joint.shape}")

    SEP = "=" * 70
    counts = np.bincount(y, minlength=2)
    print(f"\n{SEP}")
    print("TONG KET")
    print(SEP)
    print(f"  Sequences: {len(seq_meta)}  (loi/bo: {err_count})")
    print(f"  Samples  : {len(y)}  (Fall={counts[1]}, NF={counts[0]})")
    print()
    print("  Keys (custom_all.npz):")
    print(f"    X_joint, X_motion, X_angle : {X_joint.shape}")
    print("    y        : (N,) int64   0=non_fall 1=fall")
    print("    img_w    : (N,) int32")
    print("    img_h    : (N,) int32")
    print("    seq_id   : (N,) int32")
    print("    subject  : (N,) str     all")
    print("    is_flip  : (N,) int32   1=flip video")
    print("    category : (N,) str     Lying_Falls|Sitting_Falls|Standing_Falls|non_fall")
    print()
    print("  Cach dung:")
    print("    data = np.load('custom_all.npz', allow_pickle=True)")
    print(SEP)
    print("Done!")


if __name__ == "__main__":
    main()
