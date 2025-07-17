"""
Core sign language detection engine.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Any
from dataclasses import dataclass
from config import Config

class MediaPipeProcessor:
    """MediaPipe landmark extraction processor."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=config.mediapipe.min_detection_confidence,
            min_tracking_confidence=config.mediapipe.min_tracking_confidence,
            static_image_mode=config.mediapipe.static_image_mode,
            model_complexity=config.mediapipe.model_complexity
        )
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Process frame with MediaPipe.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, results)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Process with MediaPipe
        results = self.holistic.process(frame_rgb)
        
        # Convert back to BGR
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return frame_bgr, results
    
    def extract_keypoints(self, results: Any) -> np.ndarray:
        """
        Extract keypoints from MediaPipe results.
        
        Args:
            results: MediaPipe results
            
        Returns:
            Keypoints array ()
        """
        # Pose keypoints (33 points, 2 coordinates)
        pose = np.array([
            [res.x, res.y] for res in results.pose_landmarks.landmark
        ]) if results.pose_landmarks else np.zeros((33, 2))
    
        # Hand keypoints (21 points, 2 coordinates each, 2 hands)
        # important_hand_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        important_hand_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        # Left hand
        if results.left_hand_landmarks:
            lh = np.array([
                [results.left_hand_landmarks.landmark[idx].x,
                 results.left_hand_landmarks.landmark[idx].y]
                for idx in important_hand_indices
            ])
        else:
            lh = np.zeros((len(important_hand_indices), 2))
        
        # Right hand
        if results.right_hand_landmarks:
            rh = np.array([
                [results.right_hand_landmarks.landmark[idx].x,
                 results.right_hand_landmarks.landmark[idx].y]
                for idx in important_hand_indices
            ])
        else:
            rh = np.zeros((len(important_hand_indices), 2))
        # print(f"Pose shape: {pose.shape}, Left hand shape: {lh.shape}, Right hand shape: {rh.shape}")
        return np.concatenate([pose, lh, rh])
    
    def draw_landmarks(self, image: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw landmarks on image.
        
        Args:
            image: Input image
            results: MediaPipe results
            
        Returns:
            Image with landmarks drawn
        """
        # Face landmarks
        # if results.face_landmarks:
        #     self.mp_drawing.draw_landmarks(
        #         image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
        #         self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        #         self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        #     )
        
        # Pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=1)
            )
        
        # Hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1)
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
            )
        
        return image
    
    def cleanup(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'holistic'):
            self.holistic.close()
        self.logger.info("MediaPipe processor cleaned up")