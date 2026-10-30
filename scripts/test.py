import cv2
import numpy as np

def raw16_to_jpg(raw_path, width=2560, height=1984, output=None):
    with open(raw_path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint16)
    img = raw.reshape(height, width).astype(np.float32)
    
    # 截断到 10 位有效范围（0-1023）
    img = np.clip(img, 0, 1023)
    
    # 归一化到 0-255
    img_8bit = (img / 1023 * 255).astype(np.uint8)
    
    # Bayer 解码（如果你的图像颜色异常，尝试其他排列）
    img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_BayerGB2RGB)
    
    if output is None:
        output = raw_path.replace('.raw', '.jpg')
    cv2.imwrite(output, img_rgb)
    print(f"Saved to {output}")

if __name__ == "__main__":
    raw16_to_jpg('scripts/left.raw')