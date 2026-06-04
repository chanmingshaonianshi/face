"""
single_embedding.py
供 MATLAB GUI 调用：输入图片路径，输出 512 维 ArcFace embedding（逗号分隔）。
用法：python single_embedding.py "图片绝对路径"
"""
import sys, os, numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))

def imread_cn(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def main():
    if len(sys.argv) < 2:
        print("Usage: python single_embedding.py <image_path>", file=sys.stderr)
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = imread_cn(img_path)
    if img is None:
        print(f"Cannot read image: {img_path}", file=sys.stderr)
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

    print(','.join(f'{v:.8f}' for v in emb))

if __name__ == '__main__':
    main()
