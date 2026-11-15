"""
generate_samples.py -- creates 3 realistic synthetic crowd scene images
for public/samples/ (stadium.jpg, transit.jpg, plaza.jpg).
Run from project root: python ml_pipeline/generate_samples.py
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "public" / "samples"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(2024)

def draw_person(img, cx, cy, scale=1.0, skin=None, shirt=None):
    h, w = img.shape[:2]
    head_r = max(3, int(8 * scale))
    body_w  = max(4, int(10 * scale))
    body_h  = max(8, int(18 * scale))

    if skin is None:
        skin_opts = [(220,180,140),(200,150,110),(160,110,70),(100,65,40)]
        skin = skin_opts[RNG.integers(len(skin_opts))]
    if shirt is None:
        shirt = tuple(int(c) for c in RNG.integers(30, 220, 3))

    # Body
    x1, y1 = int(cx - body_w//2), int(cy)
    x2, y2 = int(cx + body_w//2), int(cy + body_h)
    if 0 <= x1 < w and 0 <= y1 < h:
        cv2.rectangle(img, (x1, y1), (min(x2,w-1), min(y2,h-1)), shirt, -1)
    # Head
    hx, hy = int(cx), int(cy - head_r)
    if 0 <= hx < w and 0 <= hy < h:
        cv2.circle(img, (hx, hy), head_r, skin, -1)
        cv2.circle(img, (hx, hy), head_r, (40,30,20), 1)

def sky_gradient(img, sky_col, ground_col):
    h, w = img.shape[:2]
    for row in range(h):
        t = row / h
        b = int(sky_col[0]*(1-t) + ground_col[0]*t)
        g = int(sky_col[1]*(1-t) + ground_col[1]*t)
        r = int(sky_col[2]*(1-t) + ground_col[2]*t)
        img[row,:] = (b,g,r)

# --- 1. STADIUM -----------------------------------------------------------
def make_stadium(path, W=800, H=500):
    img = np.zeros((H,W,3), dtype=np.uint8)
    # Sky
    sky_gradient(img, (200,170,120), (120,100,70))
    # Stands (tiered bleachers)
    for tier in range(5):
        tier_y = H//2 - tier*35
        tier_h = 30
        col_l  = max(0, 180 - tier*15)
        cv2.rectangle(img, (0,tier_y), (W,tier_y+tier_h), (col_l,col_l-10,col_l-20), -1)
    # Field
    cv2.rectangle(img, (0,H//2), (W,H), (34,139,34), -1)
    cv2.rectangle(img, (10,H//2+10),(W-10,H-10),(50,160,50),2)
    cv2.line(img,(W//2,H//2+10),(W//2,H-10),(50,180,50),2)

    # Crowd in stands - dense, clustered
    n = 320
    for i in range(n):
        tier = RNG.integers(5)
        ty   = H//2 - tier*35 + RNG.integers(5,25)
        tx   = int(RNG.uniform(20, W-20))
        scale = 0.5 + tier*0.07
        shirt_opts = [(30,30,180),(200,30,30),(30,200,30),(200,200,30),(200,100,30)]
        draw_person(img, tx, ty, scale=scale, shirt=shirt_opts[RNG.integers(5)])

    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("  -> " + str(path))

# --- 2. TRANSIT HUB -------------------------------------------------------
def make_transit(path, W=800, H=500):
    img = np.zeros((H,W,3), dtype=np.uint8)
    # Ceiling / hall
    sky_gradient(img, (220,210,200),(170,160,150))
    # Floor tiles
    for col in range(0, W, 60):
        cv2.line(img,(col,H//2),(col,H),(180,170,160),1)
    for row in range(H//2, H, 60):
        cv2.line(img,(0,row),(W,row),(180,170,160),1)
    # Pillars
    for px in [100,250,400,550,700]:
        cv2.rectangle(img,(px,H//3),(px+20,H),(100,95,90),-1)
    # Signs
    for sx in [50, 300, 600]:
        cv2.rectangle(img,(sx,30),(sx+150,70),(0,120,200),-1)
        cv2.putText(img,"METRO",(sx+10,58),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    # Dense crowd streaming through
    n = 250
    for _ in range(n):
        ty = int(RNG.uniform(H//2, H-30))
        tx = int(RNG.uniform(10, W-10))
        depth = (ty - H//2) / (H//2)
        scale = 0.5 + depth*0.7
        draw_person(img, tx, ty, scale=scale)

    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("  -> " + str(path))

# --- 3. CITY PLAZA --------------------------------------------------------
def make_plaza(path, W=800, H=500):
    img = np.zeros((H,W,3), dtype=np.uint8)
    # Sky blue
    sky_gradient(img, (180,220,240),(100,180,100))
    # Ground / plaza paving
    cv2.rectangle(img,(0,H//2),(W,H),(160,155,145),-1)
    for cx in range(0,W,80):
        for cy in range(H//2,H,80):
            cv2.rectangle(img,(cx,cy),(cx+78,cy+78),(165,160,150),1)
    # Buildings in background
    bld_data = [(20,100,120,H//2),(150,80,110,H//2),(300,60,90,H//2),
                (450,70,80,H//2),(600,90,100,H//2),(700,50,80,H//2)]
    for (bx,bw,bh,by) in bld_data:
        bld_color = (int(80+RNG.integers(20)), int(80+RNG.integers(20)), int(80+RNG.integers(40)))
        cv2.rectangle(img,(bx,by-bh),(bx+bw,by), bld_color,-1)
        for wy in range(by-bh+10,by-5,15):
            for wx2 in range(bx+8,bx+bw-5,12):
                col = (220,210,180) if RNG.random()<0.4 else (40,40,60)
                cv2.rectangle(img,(wx2,wy),(wx2+8,wy+10),col,-1)
    # Fountain
    cv2.circle(img,(W//2,H//2+40),35,(100,180,220),-1)
    cv2.circle(img,(W//2,H//2+40),35,(80,160,200),2)
    # Crowd scattered in plaza
    n = 180
    for _ in range(n):
        ty = int(RNG.uniform(H//2+5, H-20))
        tx = int(RNG.uniform(10, W-10))
        depth = (ty - H//2) / (H//2)
        scale = 0.4 + depth*0.9
        draw_person(img, tx, ty, scale=scale)

    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("  -> " + str(path))

if __name__ == "__main__":
    print("Generating sample crowd images ...")
    make_stadium(OUT / "stadium.jpg")
    make_transit(OUT / "transit.jpg")
    make_plaza(OUT / "plaza.jpg")
    print("Done OK")
