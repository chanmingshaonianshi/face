"""
single_embedding.py
供 MATLAB GUI 调用：输入图片路径，输出 512 维 ArcFace embedding。
用法：python single_embedding.py "图片绝对路径" "输出临时文件路径"

stdout 只输出 embedding 数值，所有日志/警告都走 stderr。
同时将 embedding 写入第二个参数指定的文件，作为冗余备份。
"""
import sys, os, warnings, logging, numpy as np, cv2

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

def imread_cn(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def main():
    if len(sys.argv) < 3:
        print("Usage: python single_embedding.py <image_path> <output_file>", file=sys.stderr)
        sys.exit(1)

    img_path = sys.argv[1]
    out_path = sys.argv[2]

    if not os.path.isfile(img_path):
        # 写错误标记到输出文件
        with open(out_path, 'w') as f:
            f.write('ERROR')
        sys.exit(1)

    # 加载模型期间把 stdout 重定向到 stderr
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    sys.stdout = _real_stdout

    img = imread_cn(img_path)
    if img is None:
        with open(out_path, 'w') as f:
            f.write('ERROR')
        sys.exit(1)

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

    # 写入文件（冗余备份，MATLAB 主要靠这个）
    with open(out_path, 'w') as f:
        f.write(emb_str)

    # stdout 也输出一份
    print(emb_str)

if __name__ == '__main__':
    main()
