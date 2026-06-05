"""
embedding_server.py
持久化 Python 进程：通过文件方式接收图片路径，输出 embedding 到文件。
模型只加载一次，解决摄像头每帧 1-2s 延迟问题。

协议（文件方式，避免 stdin 中文编码问题）：
  MATLAB 将图片路径写入 _server_req.txt
  Python 轮询 _server_req.txt，处理后将结果写入 _server_rep.txt
  结束: MATLAB 创建 _server_quit.lock 文件
"""
import sys, os, time, warnings, logging, numpy as np, cv2

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

def imread_cn(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(base_dir, '_server_req.txt')
    rep_file = os.path.join(base_dir, '_server_rep.txt')
    quit_lock = os.path.join(base_dir, '_server_quit.lock')
    ready_lock = os.path.join(base_dir, '_server_ready.lock')

    # 清理旧文件
    for f in [req_file, rep_file, quit_lock, ready_lock]:
        if os.path.exists(f):
            os.remove(f)

    # 模型加载
    _real_stdout = sys.stdout
    _devnull = open(os.devnull, 'w')
    sys.stdout = _devnull

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    sys.stdout = _real_stdout
    _devnull.close()

    # 通知就绪
    with open(ready_lock, 'w') as f:
        f.write('ready')

    # 主循环
    while True:
        if os.path.exists(quit_lock):
            break

        if not os.path.isfile(req_file):
            time.sleep(0.02)
            continue

        # 读取请求并删除
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                img_path = f.read().strip()
            os.remove(req_file)
        except Exception:
            time.sleep(0.02)
            continue

        if not img_path:
            time.sleep(0.02)
            continue

        # 处理图片
        reply = ''
        if not os.path.isfile(img_path):
            reply = f'error,file not found: {img_path}'
        else:
            try:
                img = imread_cn(img_path)
                if img is None:
                    reply = f'error,cannot read: {img_path}'
                else:
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
                    reply = f'ok,{emb_str}'
            except Exception:
                reply = 'error,frame processing exception'

        # 写入回复
        try:
            with open(rep_file, 'w', encoding='utf-8') as f:
                f.write(reply)
        except Exception:
            pass

        time.sleep(0.02)

    # 清理
    for f in [req_file, rep_file, quit_lock, ready_lock]:
        try:
            if os.path.exists(f): os.remove(f)
        except Exception:
            pass

if __name__ == '__main__':
    main()
