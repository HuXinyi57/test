import sys
print(f"Python版本: {sys.version}")

try:
    import cv2
    print("✓ OpenCV 已安装")
except ImportError as e:
    print(f"✗ OpenCV 安装失败: {e}")

try:
    import mediapipe as mp
    print("✓ MediaPipe 已安装")
except ImportError as e:
    print(f"✗ MediaPipe 安装失败: {e}")

try:
    import pygame
    print("✓ Pygame 已安装")
except ImportError as e:
    print(f"✗ Pygame 安装失败: {e}")

try:
    import numpy as np
    print("✓ NumPy 已安装")
except ImportError as e:
    print(f"✗ NumPy 安装失败: {e}")

print("\n所有依赖项检查完成！")