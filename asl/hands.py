"""MediaPipe hand detector: BGR camera input, clipped image coordinates."""
import cv2

class HandDetector:
    def __init__(self,max_hands=1,min_confidence=0.6):
        import mediapipe as mp
        if not hasattr(mp,"solutions"):
            raise RuntimeError("Install a MediaPipe release exposing solutions.hands")
        self.detector=mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=max_hands,min_detection_confidence=min_confidence,min_tracking_confidence=0.5)
    def detect(self,frame):
        detected=self.detector.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        boxes=[]
        for hand in detected.multi_hand_landmarks or []:
            height,width=frame.shape[:2]
            xs=[p.x for p in hand.landmark]
            ys=[p.y for p in hand.landmark]
            boxes.append((int(min(xs)*width),int(min(ys)*height),int(max(xs)*width),int(max(ys)*height)))
        return boxes
    def close(self):
        self.detector.close()
