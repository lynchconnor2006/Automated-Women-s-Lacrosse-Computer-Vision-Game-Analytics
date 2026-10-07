"""
SETUP ONLY — Calibration & Quarter Timestamps
=============================================
Run this locally BEFORE running tracker_gpu.ipynb on Colab.

All it does:
  1. Calibration  — click 8 field points, saves .calibration.json
  2. Quarter timestamps — confirm Q1-Q4 boundaries, saves .quarter_frames.json

Skips any video that already has both files.
Takes ~5 minutes per video.
No tracking, no editor, no render.

After this finishes:
  Upload all .calibration.json and .quarter_frames.json files to Google Drive
  then run tracker_gpu.ipynb on Colab.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIG — same videos as pipeline_cpu.py
# ============================================================

VIDEOS = [
    {'label': 'SYR UNC MVC Side',
     'video_path': r'C:\Users\lynch\Downloads\SYR UNC MVC Side.mp4',
     'camera': 'left'},
    {'label': 'SYR UNC LR Side',
     'video_path': r'C:\Users\lynch\Downloads\SYR UNC LR Side.mp4',
     'camera': 'right'},
]

DISP_W = 1280
DISP_H = 720

POINT_NAMES = [
    'P1: Goal Line - Left',    'P2: Goal Line - Right',
    'P3: Key Top - Left',      'P4: Key Top - Right',
    'P5: 8-Yard Line - Left',  'P6: 8-Yard Line - Right',
    'P7: 30-Yard Line - Left', 'P8: 30-Yard Line - Right',
]


# ============================================================
# FIELD LINE DRAWING (for calibration preview)
# ============================================================

def draw_field_lines(img, points, sx, sy):
    n  = len(points)
    dw = img.shape[1]; dh = img.shape[0]

    def d(p):
        return (max(0,min(int(p[0]/sx),dw-1)), max(0,min(int(p[1]/sy),dh-1)))

    def extend_line(pa, pb):
        pa,pb = d(pa),d(pb)
        if pa[0]==pb[0]: return (pa[0],0),(pa[0],dh-1)
        m = (pb[1]-pa[1])/(pb[0]-pa[0])
        return (0,int(pa[1]+m*(0-pa[0]))),(dw-1,int(pa[1]+m*(dw-1-pa[0])))

    if n>=2:
        a,b=extend_line(points[0],points[1]); cv2.line(img,a,b,(0,255,255),2)
        cv2.putText(img,'Goal',(d(points[0])[0]+5,max(d(points[0])[1]-5,10)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
    if n>=3: cv2.line(img,d(points[0]),d(points[2]),(0,255,0),2)
    if n>=4:
        cv2.line(img,d(points[1]),d(points[3]),(0,255,0),2)
        p1=np.array(d(points[0]),dtype=float); p2=np.array(d(points[1]),dtype=float)
        p3=np.array(d(points[2]),dtype=float); p4=np.array(d(points[3]),dtype=float)
        gm=(p1+p2)/2; am=(p3+p4)/2; dep=am-gm
        dl=max(np.linalg.norm(dep),1.); sp=max(np.linalg.norm(p4-p3),1.)
        ctrl=am+(dep/dl)*sp*0.3; prev=None
        for k in range(61):
            t=k/60.
            cur=(int((1-t)**2*p3[0]+2*(1-t)*t*ctrl[0]+t**2*p4[0]),
                 int((1-t)**2*p3[1]+2*(1-t)*t*ctrl[1]+t**2*p4[1]))
            if prev: cv2.line(img,prev,cur,(0,255,0),2)
            prev=cur
    if n>=6:
        a,b=extend_line(points[4],points[5]); cv2.line(img,a,b,(255,255,0),2)
        cv2.putText(img,'8-Yd',(d(points[4])[0]+5,max(d(points[4])[1]-5,10)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
    if n>=8:
        a,b=extend_line(points[6],points[7]); cv2.line(img,a,b,(0,100,255),2)
        cv2.putText(img,'30-Yd [TRACKING BOUNDARY]',
                    (d(points[6])[0]+5,max(d(points[6])[1]-5,10)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,100,255),1)
    return img


# ============================================================
# CALIBRATION
# ============================================================

def run_calibration(video_path):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    print(f'\n  Calibrating: {Path(video_path).name}')
    print('  STEP 1: Scrub to a frame where the full field is visible.')
    print('  A/D=±1   ,/.=±30   ENTER=use this frame\n')

    state = {'fi': min(10000, total-1)}
    cache = {}

    def get_f(fi):
        fi = max(0, min(fi, total-1))
        if fi not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, f = cap.read()
            if ret: cache[fi] = f
        return cache.get(fi, None)

    WIN = 'Calibration'; cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    while True:
        f = get_f(state['fi'])
        if f is None: state['fi'] = 0; continue
        oh, ow = f.shape[:2]
        img = cv2.resize(f, (DISP_W, DISP_H))
        ts  = state['fi']/fps; m=int(ts//60); s=ts%60
        cv2.rectangle(img,(0,0),(DISP_W,70),(0,0,0),-1)
        cv2.putText(img,'STEP 1 — Find a frame with full field visible',
                    (10,24),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,220,255),2)
        cv2.putText(img,f'Frame {state["fi"]}  |  {m:02d}:{s:05.2f}  |  '
                    'A/D=±1   ,/.=±30   ENTER=use this frame',
                    (10,52),cv2.FONT_HERSHEY_SIMPLEX,0.50,(180,180,180),1)
        cv2.imshow(WIN, img); k = cv2.waitKey(20) & 0xFF
        if   k in (ord('a'),ord('A')): state['fi']=max(0,state['fi']-1)
        elif k in (ord('d'),ord('D')): state['fi']=min(total-1,state['fi']+1)
        elif k==ord(','): state['fi']=max(0,state['fi']-30)
        elif k==ord('.'): state['fi']=min(total-1,state['fi']+30)
        elif k in (13,ord('\r'),ord('\n')):
            print(f'  ✓ Using frame {state["fi"]} ({m:02d}:{s:05.2f})'); break
        elif k in (ord('q'),ord('Q')):
            cap.release(); cv2.destroyWindow(WIN); return None

    chosen = get_f(state['fi']); oh,ow = chosen.shape[:2]
    sx=ow/DISP_W; sy=oh/DISP_H; base=cv2.resize(chosen,(DISP_W,DISP_H))

    print(f'\n  STEP 2: Click the 8 field points ON the visible markings.')
    for i,n in enumerate(POINT_NAMES,1): print(f'    {i}. {n}')
    print('  R=reset   S=save   Q=skip\n')

    points = []

    def draw():
        img = base.copy(); draw_field_lines(img, points, sx, sy)
        for i,p in enumerate(points):
            dp=(max(0,min(int(p[0]/sx),DISP_W-1)),max(0,min(int(p[1]/sy),DISP_H-1)))
            cv2.circle(img,dp,11,(0,0,0),-1); cv2.circle(img,dp,9,(0,255,0),-1)
            cv2.putText(img,f'P{i+1}',(dp[0]+13,dp[1]+6),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(img,f'P{i+1}',(dp[0]+12,dp[1]+5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),1)
        # Header bar at BOTTOM so it never blocks field markings
        cv2.rectangle(img,(0,DISP_H-70),(DISP_W,DISP_H),(0,0,0),-1)
        if len(points)<8:
            cv2.putText(img,f'STEP 2 — Click:  {POINT_NAMES[len(points)]}',
                        (10,DISP_H-44),cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,180,255),2)
            cv2.putText(img,f'({len(points)}/8 placed)   Place ON the field line   R=reset',
                        (10,DISP_H-18),cv2.FONT_HERSHEY_SIMPLEX,0.48,(180,180,180),1)
        else:
            cv2.putText(img,'All 8 points placed — press S to save',
                        (10,DISP_H-44),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,220,0),2)
            cv2.putText(img,'Check lines align with field markings — R=reset if not',
                        (10,DISP_H-18),cv2.FONT_HERSHEY_SIMPLEX,0.48,(180,180,180),1)
        # Point legend top-right — small and out of the way
        for i,name in enumerate(POINT_NAMES):
            color=(0,255,0) if i<len(points) else (0,220,255) if i==len(points) else (80,80,80)
            prefix='✓' if i<len(points) else ('→' if i==len(points) else '  ')
            cv2.putText(img,f'{prefix} {name}',(DISP_W-320,16+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.32,color,1)
        return img

    def on_mouse(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and len(points)<8:
            ox,oy=int(x*sx),int(y*sy); points.append((ox,oy))
            print(f'  ✓ P{len(points)} {POINT_NAMES[len(points)-1]}: ({ox},{oy})')

    cv2.setMouseCallback(WIN,on_mouse); result=None
    while True:
        cv2.imshow(WIN,draw()); k=cv2.waitKey(20)&0xFF
        if k in (ord('r'),ord('R')): points.clear(); print('  Reset.')
        elif k in (ord('s'),ord('S')):
            if len(points)==8:
                cv2.destroyWindow(WIN)
                cal={'points':points,'point_names':POINT_NAMES,
                     'video_path':str(video_path),'frame_width':ow,'frame_height':oh,
                     'calibration_frame':state['fi'],'timestamp':datetime.now().isoformat()}
                out=str(Path(video_path).with_suffix('.calibration.json'))
                with open(out,'w') as f: json.dump(cal,f,indent=2)
                print(f'  ✓ Saved → {out}'); result=cal; break
            else: print(f'  Only {len(points)}/8 — keep clicking.')
        elif k in (ord('q'),ord('Q')): cv2.destroyWindow(WIN); break

    cap.release(); return result


# ============================================================
# QUARTER TIMESTAMPS
# ============================================================

def run_quarter_timestamps(video_path, fps, total_frames):
    cap = cv2.VideoCapture(video_path)

    def parse(s):
        try:
            p=s.strip().split(':')
            return int(p[0])*60+float(p[1]) if len(p)==2 else float(p[0])
        except: return None

    def secs_to_f(s): return max(0,min(int(s*fps),total_frames-1))

    print(f'\n  Quarter timestamps: {Path(video_path).name}')
    print(f'  {total_frames:,} frames | {fps:.1f} fps | {total_frames/fps/60:.1f} min')
    print(f'  Enter MM:SS for each boundary.\n')

    raw = {}
    for q in range(1,5):
        while True:
            s=input(f'  Q{q} START (MM:SS): '); e=input(f'  Q{q} END   (MM:SS): ')
            ss,es=parse(s),parse(e)
            if ss is not None and es is not None and es>ss: raw[q]=(ss,es); break
            print('  Invalid — try again.')

    confirmed={}; cache={}

    def get_f(fi):
        fi=max(0,min(fi,total_frames-1))
        if fi not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES,fi); ret,f=cap.read()
            if ret: cache[fi]=cv2.resize(f,(DISP_W,DISP_H))
        return cache.get(fi,np.zeros((DISP_H,DISP_W,3),np.uint8))

    for q in range(1,5):
        for boundary,raw_sec in [('START',raw[q][0]),('END',raw[q][1])]:
            fi=secs_to_f(raw_sec); state={'fi':fi}
            WIN=f'Q{q}_{boundary}'; cv2.namedWindow(WIN,cv2.WINDOW_AUTOSIZE)
            print(f'\n  Confirm Q{q} {boundary} — A/D=±1  ,/.=±30  ENTER=confirm')
            while True:
                img=get_f(state['fi']).copy()
                ts=state['fi']/fps; m=int(ts//60); s=ts%60
                cv2.rectangle(img,(0,0),(DISP_W,80),(0,0,0),-1)
                cv2.putText(img,f'Q{q} {boundary} | Frame {state["fi"]} | {m:02d}:{s:05.2f}',
                            (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,220,255),2)
                cv2.putText(img,'A/D=±1   ,/.=±30   ENTER=confirm',
                            (10,58),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
                cv2.putText(img,f'Should show Q{q} {boundary.lower()} (faceoff / whistle)',
                            (10,76),cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,255,128),1)
                cv2.imshow(WIN,img); k=cv2.waitKey(20)&0xFF
                if   k in (ord('a'),ord('A')): state['fi']=max(0,state['fi']-1)
                elif k in (ord('d'),ord('D')): state['fi']=min(total_frames-1,state['fi']+1)
                elif k==ord(','): state['fi']=max(0,state['fi']-30)
                elif k==ord('.'): state['fi']=min(total_frames-1,state['fi']+30)
                elif k in (13,ord('\r'),ord('\n')):
                    print(f'  ✓ Q{q} {boundary}: frame {state["fi"]} ({state["fi"]/fps:.1f}s)')
                    cv2.destroyWindow(WIN); break
            if q not in confirmed: confirmed[q]={}
            confirmed[q][boundary]=state['fi']

    cap.release()
    qf={q:(confirmed[q]['START'],confirmed[q]['END']) for q in range(1,5)}
    print(f'\n  Quarter windows:')
    for q,(s,e) in qf.items():
        print(f'    Q{q}: frame {s:,} → {e:,}  ({(e-s)/fps:.0f}s)')
    return qf


# ============================================================
# MAIN
# ============================================================

def main():
    print('='*60)
    print('  SETUP ONLY — Calibration & Quarter Timestamps')
    print('  No tracking will run. Takes ~5 min per video.')
    print('='*60)

    for cfg in VIDEOS:
        label      = cfg['label']
        video_path = cfg['video_path']
        cal_path   = Path(video_path).with_suffix('.calibration.json')
        qf_path    = Path(video_path).with_suffix('.quarter_frames.json')

        print(f"\n{'='*60}\n  {label}\n{'='*60}")

        if not Path(video_path).exists():
            print(f'  ✗ Video not found: {video_path}')
            print(f'  Skipping.')
            continue

        # ── Calibration ───────────────────────────────────────
        if cal_path.exists():
            print(f'  ✓ Calibration already exists — skipping')
            print(f'    {cal_path}')
        else:
            cal = run_calibration(video_path)
            if cal is None:
                print(f'  Calibration skipped — re-run setup_only.py to redo')
                continue

        # ── Quarter timestamps ────────────────────────────────
        if qf_path.exists():
            print(f'  ✓ Quarter frames already exist — skipping')
            print(f'    {qf_path}')
        else:
            cap_tmp = cv2.VideoCapture(video_path)
            fps     = cap_tmp.get(cv2.CAP_PROP_FPS)
            total   = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_tmp.release()
            qf = run_quarter_timestamps(video_path, fps, total)
            with open(qf_path,'w') as f:
                json.dump({str(k):list(v) for k,v in qf.items()},f,indent=2)
            print(f'  ✓ Saved → {qf_path}')

        print(f'\n  ✓ {label} setup complete')
        print(f'    Upload these to Google Drive (My Drive/Lacrosse/):')
        print(f'      {cal_path.name}')
        print(f'      {qf_path.name}')

    print(f'\n{"="*60}')
    print('  ALL SETUP COMPLETE')
    print('='*60)
    print('\n  Next steps:')
    print('  1. Upload all .calibration.json and .quarter_frames.json')
    print('     files to My Drive/Lacrosse/ in Google Drive')
    print('  2. Run tracker_gpu.ipynb on Colab (one video at a time)')
    print('  3. Download the .tracking_cache.json files from Drive')
    print('  4. Place them in your local output folder')
    print('  5. Run pipeline_cpu.py for team setup, editing, and export')


if __name__ == '__main__':
    main()