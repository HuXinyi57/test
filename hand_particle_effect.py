import cv2
import mediapipe as mp
import numpy as np
import random

class Particle:
    def __init__(self, x, y, color, size=3):
        self.x = x
        self.y = y
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-3, 3)
        self.color = color
        self.size = size
        self.life = 1.0
        self.decay = random.uniform(0.015, 0.04)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.08
        self.life -= self.decay
        return self.life > 0

    def draw(self, frame):
        alpha = int(self.life * 255)
        radius = int(self.size * self.life)
        if radius > 0 and self.life > 0:
            overlay = frame.copy()
            cv2.circle(overlay, (int(self.x), int(self.y)), radius, self.color, -1)
            cv2.addWeighted(overlay, self.life, frame, 1 - self.life, 0, frame)

class HandParticleEffect:
    def __init__(self):
        self.screen_width = 1280
        self.screen_height = 720

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_height)

        self.particles = []
        self.finger_colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255)
        ]

    def get_finger_tips(self, results):
        tips = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in [4, 8, 12, 16, 20]:
                    x = hand_landmarks.landmark[i].x * self.screen_width
                    y = hand_landmarks.landmark[i].y * self.screen_height
                    tips.append((x, y, i // 4 - 1))
        return tips

    def run(self):
        print("手势粒子效果程序启动！")
        print("按 'q' 键退出")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                finger_tips = self.get_finger_tips(results)
                for x, y, finger_idx in finger_tips:
                    for _ in range(2):
                        color = self.finger_colors[finger_idx]
                        self.particles.append(Particle(x, y, color))

                self.particles = [p for p in self.particles if p.update()]

                for particle in self.particles:
                    particle.draw(frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                        )

                cv2.imshow('手势粒子效果', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("程序已退出")

if __name__ == "__main__":
    effect = HandParticleEffect()
    effect.run()