import time,csv,cv2,threading,numpy as np,pycuda.autoinit,pycuda.driver as cuda,tensorrt as trt
from pathlib import Path
from scipy.ndimage import maximum_filter

class TegraMonitor:
    """后台线程定期运行 tegrastats 并解析资源指标。"""
    _RE = (
        r"RAM (?P<ram_used>\d+)/(?P<ram_total>\d+)MB"
        r".*?SWAP (?P<swap_used>\d+)/(?P<swap_total>\d+)MB"
        r".*?CPU \[(?P<cpu_cores>[^\]]+)\]"
        r".*?GR3D_FREQ (?P<gpu_util>\d+)%"
        r".*?CPU@(?P<cpu_temp>[\d.]+)C"
        r".*?GPU@(?P<gpu_temp>[\d.]+)C"
    )

    def __init__(self, interval=1.0):
        self.interval = interval
        self._stop = threading.Event()
        self._samples = []
        self._thread = None

    def start(self):
        import subprocess
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(int(self.interval * 1000))],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True, bufsize=1,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import re
        pat = re.compile(self._RE)
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = pat.search(line)
            if not m:
                continue
            cpu_str = m.group("cpu_cores")
            cores = [int(s.split("%")[0]) for s in cpu_str.split(",")]
            self._samples.append({
                "timestamp": time.time(),
                "cpu_total_util_%": round(sum(cores) / len(cores), 2) if cores else 0,
                "cpu_cores_util_%": ",".join(str(c) for c in cores),
                "gpu_util_%": int(m.group("gpu_util")),
                "ram_used_MB": int(m.group("ram_used")),
                "ram_total_MB": int(m.group("ram_total")),
                "swap_used_MB": int(m.group("swap_used")),
                "cpu_temp_C": float(m.group("cpu_temp")),
                "gpu_temp_C": float(m.group("gpu_temp")),
            })
        self._proc.stdout.close()

    def stop(self, csv_path):
        self._stop.set()
        self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=5)
        if not self._samples:
            return
        keys = list(self._samples[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self._samples)
        print(f"tegrastats saved: {csv_path} ({len(self._samples)} samples)")

IMG_DIR="images"
LABEL_DIR="result"
SAVE_VIS=False
ENGINE_PATH="models/p234_fp16_1024_b1.engine"
OUTPUT_LAYOUT="center_wh_heatmap"
PATCH=1024
STRIDE=4
NUM_CLASSES=3
STEP=int(PATCH*0.95)
PEAK_KERNEL=5
MIN_CENTER_DIST=12
IMAGE_EXT=".JPG"

def sigmoid(x): return 1/(1+np.exp(-x))

def preprocess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    return np.transpose(img,(2,0,1))[None]

def split_pred(pred):
    if OUTPUT_LAYOUT=="heatmap_center_wh":
        hm_logits=pred[0:NUM_CLASSES]
        center=pred[NUM_CLASSES:NUM_CLASSES+2]
        wh=pred[NUM_CLASSES+2:NUM_CLASSES+4]
    elif OUTPUT_LAYOUT=="center_wh_heatmap":
        center=pred[0:2]
        wh=pred[2:4]
        hm_logits=pred[4:4+NUM_CLASSES]
    else:
        raise ValueError(f"unknown OUTPUT_LAYOUT: {OUTPUT_LAYOUT}")
    return hm_logits,center,wh

def sliding_pos(length):
    if length<=PATCH: return [0]
    pos=list(range(0,length-PATCH+1,STEP))
    if pos[-1]!=length-PATCH: pos.append(length-PATCH)
    return pos

class TRTInfer:
    def __init__(self,engine_path):
        logger=trt.Logger(trt.Logger.WARNING)
        print("loading engine:",engine_path)
        with open(engine_path,"rb") as f:
            runtime=trt.Runtime(logger)
            engine=runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"failed to load engine: {engine_path}")
        self.context=engine.create_execution_context()
        self.host=[None]*engine.num_bindings
        self.dev=[None]*engine.num_bindings
        self.bindings=[0]*engine.num_bindings
        self.in_id=None
        self.out_ids=[]
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                self.in_id=i
                self.context.set_binding_shape(i,(1,3,PATCH,PATCH))
        for i in range(engine.num_bindings):
            shape=tuple(self.context.get_binding_shape(i))
            dtype=trt.nptype(engine.get_binding_dtype(i))
            h=np.empty(shape,dtype=dtype)
            d=cuda.mem_alloc(h.nbytes)
            self.host[i]=h
            self.dev[i]=d
            self.bindings[i]=int(d)
            if not engine.binding_is_input(i): self.out_ids.append(i)
        print("engine loaded, outputs:",len(self.out_ids))
    def infer(self,x):
        np.copyto(self.host[self.in_id],x)
        cuda.memcpy_htod(self.dev[self.in_id],self.host[self.in_id])
        self.context.execute_v2(self.bindings)
        cuda.Context.synchronize()
        outs=[]
        for i in self.out_ids:
            cuda.memcpy_dtoh(self.host[i],self.dev[i])
            outs.append(self.host[i].astype(np.float32))
        return outs

def decode_boxes(heatmap,center,wh,stride,img_h,img_w):
    threshold=0.3
    candidates=[]
    for cls in range(NUM_CLASSES):
        hm=heatmap[cls]
        peaks=(hm==maximum_filter(hm,size=PEAK_KERNEL))&(hm>threshold)
        ys,xs=np.where(peaks)
        for y,x in zip(ys,xs):
            dx,dy=center[:,y,x]
            w_box,h_box=wh[:,y,x]
            if w_box<=0 or h_box<=0:
                continue
            cx=(x+dx)*stride
            cy=(y+dy)*stride
            w_box*=stride
            h_box*=stride
            x1=max(0,int(cx-w_box/2))
            y1=max(0,int(cy-h_box/2))
            x2=min(img_w,int(cx+w_box/2))
            y2=min(img_h,int(cy+h_box/2))
            if x2<=x1 or y2<=y1:
                continue
            score=float(hm[y,x])
            candidates.append([x1,y1,x2,y2,cls,score,cx,cy])
    kept=[]
    for cls in range(NUM_CLASSES):
        cls_boxes=[b for b in candidates if b[4]==cls]
        cls_boxes.sort(key=lambda x:x[5],reverse=True)
        grid={}
        for b in cls_boxes:
            cx,cy=b[6],b[7]
            gx=int(cx//MIN_CENTER_DIST); gy=int(cy//MIN_CENTER_DIST)
            duplicate=False
            for nx in (gx-1,gx,gx+1):
                for ny in (gy-1,gy,gy+1):
                    for kx,ky in grid.get((nx,ny),()):
                        if (cx-kx)**2+(cy-ky)**2 < MIN_CENTER_DIST**2:
                            duplicate=True; break
                    if duplicate: break
                if duplicate: break
            if not duplicate:
                kept.append(b[:6])
                grid.setdefault((gx,gy),[]).append((cx,cy))
    return kept

def list_images(img_dir):
    img_dir=Path(img_dir)
    return sorted(p for p in img_dir.iterdir() if p.name.endswith(IMAGE_EXT))

def infer_image(trt_model,img_path,out_dir):
    t0=time.time()
    img=cv2.imread(str(img_path))
    if img is None:
        print("skip unreadable:",img_path)
        return
    H,W=img.shape[:2]
    xs=sliding_pos(W); ys=sliding_pos(H)
    out_h=H//STRIDE; out_w=W//STRIDE
    heatmap_sum=np.zeros((NUM_CLASSES,out_h,out_w),np.float32)
    center_sum=np.zeros((2,out_h,out_w),np.float32)
    wh_sum=np.zeros((2,out_h,out_w),np.float32)
    count_map=np.zeros((out_h,out_w),np.float32)
    for y in ys:
        for x in xs:
            patch=img[y:y+PATCH,x:x+PATCH]
            if patch.shape[0]!=PATCH or patch.shape[1]!=PATCH:
                ph, pw = patch.shape[:2]
                patch = cv2.copyMakeBorder(patch,0,PATCH-ph,0,PATCH-pw,cv2.BORDER_CONSTANT,value=0)
            inp=preprocess(patch)
            pred=trt_model.infer(inp)[0][0]
            hm_logits,center,wh=split_pred(pred)
            oy=y//STRIDE; ox=x//STRIDE
            ph,pw=hm_logits.shape[-2:]
            vh=min(ph,out_h-oy); vw=min(pw,out_w-ox)
            if vh<=0 or vw<=0:
                continue
            heatmap_sum[:,oy:oy+vh,ox:ox+vw]+=hm_logits[:,:vh,:vw]
            center_sum[:,oy:oy+vh,ox:ox+vw]+=center[:,:vh,:vw]
            wh_sum[:,oy:oy+vh,ox:ox+vw]+=wh[:,:vh,:vw]
            count_map[oy:oy+vh,ox:ox+vw]+=1
    count_map[count_map==0]=1
    heatmap_logits=heatmap_sum/count_map[np.newaxis,:,:]
    heatmap=sigmoid(heatmap_logits)
    center=center_sum/count_map
    wh=wh_sum/count_map
    bboxes=decode_boxes(heatmap,center,wh,STRIDE,H,W)
    result=img.copy()
    colors=[(0,255,0),(255,0,0),(0,0,255)]
    label_path=out_dir/(img_path.stem+".txt")
    with open(label_path,"w") as f:
        for x1,y1,x2,y2,cls,score in bboxes:
            if SAVE_VIS:
                cv2.rectangle(result,(x1,y1),(x2,y2),colors[cls],2)
                cv2.putText(result,f"{cls}:{score:.2f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,colors[cls],2)
            cx=((x1+x2)*0.5)/W
            cy=((y1+y2)*0.5)/H
            bw=(x2-x1)/W
            bh=(y2-y1)/H
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    if SAVE_VIS:
        result_path=out_dir/(img_path.stem+".jpg")
        cv2.imwrite(str(result_path),result)
    print(f"{img_path.name}: objects={len(bboxes)} time={time.time()-t0:.3f}s")

def main():
    print("image dir:",IMG_DIR)
    print("label dir:",LABEL_DIR)
    print("save vis:",SAVE_VIS)
    print("output layout:",OUTPUT_LAYOUT)
    img_paths=list_images(IMG_DIR)
    if not img_paths:
        raise FileNotFoundError(f"no images found in {IMG_DIR}")
    out_dir=Path(LABEL_DIR)
    out_dir.mkdir(parents=True,exist_ok=True)
    print("images:",len(img_paths))
    monitor=TegraMonitor(interval=1.0)
    monitor.start()
    trt_model=TRTInfer(ENGINE_PATH)
    for i,img_path in enumerate(img_paths,1):
        print(f"[{i}/{len(img_paths)}]",end=" ")
        infer_image(trt_model,img_path,out_dir)
    monitor.stop(out_dir/"jtop_metrics.csv")

if __name__=="__main__":
    main()
