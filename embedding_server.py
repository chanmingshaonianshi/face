"""
embedding_server.py
持久化 Python 进程：通过 stdin 接收图片路径，输出 embedding 到 stdout。
MATLAB 通过 readline 通信，模型只加载一次，解决摄像头每帧 1-2s 延迟问题。

协议：
  MATLAB 写一行: "图片绝对路径\n"
  Python 读取后处理，输出一行: "ok,512个浮点数\n" 或 "error,错误信息\n"
  结束: MATLAB 发送 "quit\n" 或关闭 stdin
"""
import sys, os, warnings, logging, numpy as np, cv2

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

def imread_cn(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def main():
    # 把模型加载日志打到 stderr，保持 stdout 干净
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    sys.stdout = _real_stdout

    # 通知 MATLAB 模型就绪
    print('ready', flush=True)

    for line in sys.stdin:
        img_path = line.strip()
        if not img_path or img_path == 'quit':
            break

        if not os.path.isfile(img_path):
            print(f'error,file not found: {img_path}', flush=True)
            continue

        img = imread_cn(img_path)
        if img is None:
            print(f'error,cannot read: {img_path}', flush=True)
            continue

        faces = app.get(img)
        if len(faces) == 0:
            h, w = img.shape[:2]
            sz = min(h, w)
            cy, cx = h // 2, w // 2
            crop = img[cy - sz//2:cy + sz//2, cx - sz//2:cx + sz//2]
            face_img = cv2.resize(crop, (112, 112))
            emb = app.models['recognition'].get_feat(face_img).flatten().astype(np.float32)
        else:
            emb = faces[0].normed_embedding.astype(np.float32)

        emb_str = ','.join(f'{v:.8f}' for v in emb)
        print(f'ok,{emb_str}', flush=True)

if __name__ == '__main__':
    main()
