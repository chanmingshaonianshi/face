"""
extract_embeddings.py
用 insightface (ArcFace) 提取 train_data / test_data 所有人脸的 512 维 embedding，
保存为 .mat 文件供 MATLAB 读取。
"""
import os, sys
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

ROOT = r'c:\Users\john\OneDrive\Desktop\Git项目\face'

def extract_label(filename):
    """与 MATLAB ImagePreprocess / eval_pipeline 一致的标签提取逻辑"""
    base = os.path.splitext(filename)[0]
    # 优先取第一个下划线前的内容
    idx = base.find('_')
    if idx != -1:
        return base[:idx]
    # 兼容无下划线：去除末尾数字和连接符
    import re
    label = re.sub(r'[\d\s_\-]+$', '', base)
    return label if label else 'Unknown'

def imread_cn(path):
    """cv2.imread 在 Windows 不支持中文路径，用 imdecode 绕过"""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def list_images(folder):
    """递归列出所有图片"""
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = []
    for root, dirs, fnames in os.walk(folder):
        for f in sorted(fnames):
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(root, f))
    return files

def main():
    print("Loading insightface model (buffalo_l)...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model loaded.")

    for split in ['train_data', 'test_data']:
        folder = os.path.join(ROOT, split)
        files = list_images(folder)
        print(f"\nProcessing {split}: {len(files)} images")

        embeddings = []
        labels = []
        filenames = []
        failed = 0

        for i, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            img = imread_cn(fpath)
            if img is None:
                print(f"  [WARN] Cannot read: {fname}")
                failed += 1
                continue

            faces = app.get(img)
            if len(faces) == 0:
                # 检测器失败 → 整图强制送识别模型（直接抽 embedding，不依赖检测）
                # insightface.recognition model 期望 112x112，我们居中裁剪再 resize
                h, w = img.shape[:2]
                sz = min(h, w)
                cy, cx = h // 2, w // 2
                crop = img[cy - sz//2:cy + sz//2, cx - sz//2:cx + sz//2]
                face_img = cv2.resize(crop, (112, 112))
                # 直接用 recognition model 抽特征
                emb = app.models['recognition'].get_feat(face_img).flatten().astype(np.float32)
            else:
                emb = faces[0].normed_embedding.astype(np.float32)

            embeddings.append(emb)
            labels.append(extract_label(fname))
            filenames.append(fname)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(files)} ...")

        emb_array = np.array(embeddings, dtype=np.float32)  # N x 512
        print(f"  Done. Embeddings: {emb_array.shape}, Failed: {failed}")

        out_path = os.path.join(ROOT, f'{split}_embeddings.mat')
        # 保存为 MATLAB v7.3 兼容格式
        from scipy.io import savemat
        savemat(out_path, {
            'embeddings': emb_array,
            'labels': np.array(labels, dtype=object),
            'filenames': np.array(filenames, dtype=object),
            'failed': failed,
        })
        print(f"  Saved to {out_path}")

    print("\n=== All done. ===")

if __name__ == '__main__':
    main()
