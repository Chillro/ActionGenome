import cv2
import pandas as pd
import pickle
import os
import numpy as np
from collections import defaultdict

# ===== 共通設定 =====
ROOT = r"C:\Users\takeu\pydata\ActionGenome\dataset"
CHARADES_CSV = os.path.join(ROOT, "Charades", "Charades_annotation", "Charades_v1_train.csv")
CLASSES_TXT = os.path.join(ROOT, "Charades", "Charades_annotation", "Charades_v1_classes.txt")
AG_PERSON = os.path.join(ROOT, "ActionGenome", "annotations", "person_bbox.pkl")
AG_OBJECT = os.path.join(ROOT, "ActionGenome", "annotations", "object_bbox_and_relationship.pkl")
VIDEO_DIR = os.path.join(ROOT, "Charades", "Charades_v1_480")
OUT_DIR = r"C:\Users\takeu\pydata\ActionGenome\outputs\annotated_videos"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== Charades Action ID→名前 =====
id2act = {}
with open(CLASSES_TXT, "r") as f:
    for line in f:
        idx, name = line.strip().split(" ", 1)
        id2act[idx] = name

# ===== Charades 読み込み =====
df = pd.read_csv(CHARADES_CSV)

# actions が文字列のもの= train用動画
df = df[df['actions'].apply(lambda x: isinstance(x, str))]
video_ids = sorted(df['id'].unique())

def load_actions_for_video(vid):
    rows = df[df['id'] == vid]
    if len(rows) == 0:
        return []
    acts = []
    for _, row in rows.iterrows():
        acts_str = row['actions']
        if not isinstance(acts_str, str):
            continue  # testを無視
        for a in acts_str.split(';'):
            if a == '':
                continue
            cls, ts, te = a.split(' ')
            cls_name = id2act.get(cls, cls)
            acts.append((cls_name, float(ts), float(te)))
    return acts

# ===== Load ActionGenome =====
with open(AG_PERSON, 'rb') as f:
    person_raw = pickle.load(f)
with open(AG_OBJECT, 'rb') as f:
    obj_raw = pickle.load(f)

def build_db(raw, vid):
    db = defaultdict(list)
    prefix = vid + ".mp4/"
    for key, v in raw.items():
        if key.startswith(prefix):
            fname = key.split('/')[-1]
            frame = int(fname.replace('.png',''))
            db[frame].append(v)
    return db

# ===== bbox 抽出 =====
def extract_bboxes_person(entry):
    b = entry.get('bbox')
    if not isinstance(b, np.ndarray) or b.size == 0:
        return []
    out=[]
    for row in b:
        if len(row) >= 4:
            x1,y1,x2,y2=row[:4]
            if x2 > x1 and y2 > y1:
                out.append((float(x1),float(y1),float(x2),float(y2)))
    return out

def extract_bboxes_object(e):
    b = e.get('bbox')
    if b is None:
        return []
    if isinstance(b, tuple) and len(b) == 4:
        x,y,w,h=b
        if w>0 and h>0:
            return [(float(x),float(y),float(x+w),float(y+h))]
        return []
    if isinstance(b,(list,tuple)) and len(b)>=4:
        x1,y1,x2,y2=b[:4]
        if x2>x1 and y2>y1:
            return [(float(x1),float(y1),float(x2),float(y2))]
    return []

# ===== 全動画ループ（trainのみ）=====
total = len(video_ids)

for idx, VIDEO_ID in enumerate(video_ids, 1):
    print(f"\n[ {idx:5d}/{total} ] Processing {VIDEO_ID}")

    charades_actions = load_actions_for_video(VIDEO_ID)
    if len(charades_actions)==0:
        print(" No actions → skip.")
        continue

    person_db = build_db(person_raw, VIDEO_ID)
    obj_db = build_db(obj_raw, VIDEO_ID)
    if len(person_db)==0 and len(obj_db)==0:
        print(" No AG bbox → skip.")
        continue

    video_path = os.path.join(VIDEO_DIR, f"{VIDEO_ID}.mp4")
    if not os.path.exists(video_path):
        print(" No video file → skip.")
        continue

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save_path = os.path.join(OUT_DIR, f"{VIDEO_ID}_annotated.mp4")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    frame_idx = 0
    duplicate = int(0.5 * fps)
    any_frame_written = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        acts=[c for (c,ts,te) in charades_actions if ts<=t<=te]
        has_action=len(acts)>0

        person_boxes=[]
        if frame_idx in person_db:
            for entry in person_db[frame_idx]:
                person_boxes.extend(extract_bboxes_person(entry))
        has_person=len(person_boxes)>0

        object_boxes=[]
        if frame_idx in obj_db:
            for obj_entry in obj_db[frame_idx]:
                entries=obj_entry if isinstance(obj_entry,list) else [obj_entry]
                for e in entries:
                    if isinstance(e,dict):
                        object_boxes.extend(extract_bboxes_object(e))
        has_object=len(object_boxes)>0

        if not (has_action and (has_person or has_object)):
            frame_idx += 1
            continue

        for i,a in enumerate(acts):
            cv2.putText(frame,f"Act:{a}",(20,30+20*i),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        for (x1,y1,x2,y2) in person_boxes:
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(frame,"person",(int(x1),int(y1)-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        for obj_entry in obj_db.get(frame_idx,[]):
            entries=obj_entry if isinstance(obj_entry,list) else [obj_entry]
            for e in entries:
                if not isinstance(e,dict):
                    continue
                name=e.get('class','obj')
                for (x1,y1,x2,y2) in extract_bboxes_object(e):
                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
                    cv2.putText(frame,name,(int(x1),int(y1)-4),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

        for _ in range(duplicate):
            writer.write(frame)

        any_frame_written = True
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if any_frame_written:
        print(f" Output → {save_path}")
    else:
        print(" No valid frame → skip output")
