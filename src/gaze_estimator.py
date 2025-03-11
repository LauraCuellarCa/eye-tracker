import numpy as np
import cv2
from collections import deque

class GazeEstimator:
    """
    Class for estimating gaze position on screen based on eye landmarks
    This is a simplified gaze estimator that uses a calibration step
    to map eye positions to screen coordinates
    """
    def __init__(self, screen_width=1920, screen_height=1080, smoothing_factor=0.3, history_length=10):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_points = []
        self.calibration_eye_features = []
        self.is_calibrated = False
        self.mapping_model = None
        self.smoothing_factor = smoothing_factor
        self.position_history = deque(maxlen=history_length)
        self.last_position = (screen_width//2, screen_height//2)
        
    def extract_eye_features(self, eye_data):
        """
        Extract features from detected eye landmarks
        Features: pupil positions, eye corners, eye widths, eye heights
        """
        if eye_data is None:
            return None
            
        left_eye = eye_data['left_eye']
        right_eye = eye_data['right_eye']
        
        # Get pupil centers
        left_pupil = np.mean(left_eye, axis=0)
        right_pupil = np.mean(right_eye, axis=0)
        
        # Get eye corners (lateral and medial)
        left_lateral = left_eye[0]  # Leftmost point of left eye
        left_medial = left_eye[3]   # Rightmost point of left eye
        right_lateral = right_eye[3]  # Rightmost point of right eye
        right_medial = right_eye[0]   # Leftmost point of right eye
        
        # Calculate eye dimensions
        left_width = np.linalg.norm(left_lateral - left_medial)
        right_width = np.linalg.norm(right_lateral - right_medial)
        
        # Calculate vertical dimensions (simplified)
        left_height = np.linalg.norm(left_eye[1] - left_eye[5])
        right_height = np.linalg.norm(right_eye[1] - right_eye[5])
        
        # Calculate pupil positions relative to eye corners
        # This helps normalize for different face positions
        left_rel_x = (left_pupil[0] - left_lateral[0]) / left_width
        left_rel_y = (left_pupil[1] - left_eye[1][1]) / left_height
        right_rel_x = (right_pupil[0] - right_medial[0]) / right_width
        right_rel_y = (right_pupil[1] - right_eye[1][1]) / right_height
        
        # Create feature vector
        features = np.array([
            left_rel_x, left_rel_y,
            right_rel_x, right_rel_y,
            left_width, left_height,
            right_width, right_height
        ])
        
        return features
        
    def add_calibration_point(self, screen_point, eye_features):
        """
        Add a calibration point (screen coordinates and corresponding eye features)
        """
        if eye_features is None:
            return False
            
        self.calibration_points.append(screen_point)
        self.calibration_eye_features.append(eye_features)
        return True
        
    def calibrate(self):
        """
        Calibrate the gaze estimator using collected calibration points
        This method fits a simple linear regression model
        """
        if len(self.calibration_points) < 5:
            print("Not enough calibration points. At least 5 required.")
            return False
            
        # Convert to numpy arrays
        X = np.array(self.calibration_eye_features)
        y = np.array(self.calibration_points)
        
        # Using simple linear regression for demo purposes
        # In a real implementation, consider SVR or neural networks
        from sklearn.linear_model import LinearRegression
        self.mapping_model = LinearRegression()
        self.mapping_model.fit(X, y)
        self.is_calibrated = True
        return True
        
    def estimate_gaze_point(self, eye_data):
        """
        Estimate gaze point (screen coordinates) based on eye data
        """
        if not self.is_calibrated:
            return None
            
        eye_features = self.extract_eye_features(eye_data)
        if eye_features is None:
            return self.last_position
            
        # Predict gaze point
        predicted_point = self.mapping_model.predict([eye_features])[0]
        
        # Apply smoothing
        if self.last_position is not None:
            smoothed_x = self.smoothing_factor * predicted_point[0] + (1 - self.smoothing_factor) * self.last_position[0]
            smoothed_y = self.smoothing_factor * predicted_point[1] + (1 - self.smoothing_factor) * self.last_position[1]
            predicted_point = (smoothed_x, smoothed_y)
        
        # Ensure coordinates are within screen bounds
        x = max(0, min(predicted_point[0], self.screen_width))
        y = max(0, min(predicted_point[1], self.screen_height))
        
        gaze_point = (int(x), int(y))
        self.last_position = gaze_point
        self.position_history.append(gaze_point)
        
        return gaze_point
        
    def perform_calibration_sequence(self, eye_detector, num_points=9):
        """
        Interactive calibration sequence showing points on screen
        This is a placeholder - implementation would depend on your display system
        """
        # In a real implementation, you would:
        # 1. Show a series of calibration targets on screen
        # 2. For each target, collect eye features
        # 3. Add each (target, eye_features) pair to calibration data
        # 4. Call self.calibrate() at the end
        
        # For the purposes of this project, you'd implement this
        # with your UI library of choice (e.g., pygame, tkinter, etc.)
        pass 