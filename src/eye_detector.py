import cv2
import numpy as np
import mediapipe as mp
from imutils import face_utils

class EyeDetector:
    """
    Class for detecting eyes and facial landmarks using webcam feed with MediaPipe
    """
    def __init__(self):
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
            
        # Define indices for left and right eyes in MediaPipe's 468 landmarks
        # These correspond to the eye contour
        self.left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Initialize webcam
        self.cap = None
        
    def start_webcam(self, camera_id=0):
        """Start the webcam feed"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam with ID", camera_id)
        return self.cap.isOpened()
    
    def stop_webcam(self):
        """Release the webcam"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
    
    def get_frame(self):
        """Get the current frame from webcam"""
        if not self.cap or not self.cap.isOpened():
            raise ValueError("Webcam not initialized. Call start_webcam() first.")
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def detect_eyes(self, frame):
        """
        Detect eyes in a given frame using MediaPipe
        
        Returns:
            dict: Dictionary with 'left_eye' and 'right_eye' coordinates,
                 'face' rectangle, and 'landmarks' points
        """
        if frame is None:
            return None
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first detected face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to numpy array
        h, w, _ = frame.shape
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
        landmarks = np.array(landmarks)
        
        # Extract eye coordinates
        left_eye = np.array([landmarks[i] for i in self.left_eye_indices])
        right_eye = np.array([landmarks[i] for i in self.right_eye_indices])
        
        # Create a face rectangle
        # MediaPipe doesn't provide a face rectangle directly, so we'll create one
        # based on the landmarks
        x_min = np.min(landmarks[:, 0])
        y_min = np.min(landmarks[:, 1])
        x_max = np.max(landmarks[:, 0])
        y_max = np.max(landmarks[:, 1])
        
        # Create a face object with similar interface to dlib's rectangle
        class FaceRect:
            def __init__(self, left, top, right, bottom):
                self.left_val = left
                self.top_val = top
                self.right_val = right
                self.bottom_val = bottom
                
            def left(self): 
                return self.left_val
                
            def top(self): 
                return self.top_val
                
            def right(self): 
                return self.right_val
                
            def bottom(self): 
                return self.bottom_val
                
            def width(self): 
                return self.right_val - self.left_val
                
            def height(self): 
                return self.bottom_val - self.top_val
        
        face = FaceRect(x_min, y_min, x_max, y_max)
        
        return {
            'left_eye': left_eye,
            'right_eye': right_eye,
            'face': face,
            'landmarks': landmarks
        }
    
    def draw_eyes(self, frame, eye_data):
        """Draw eye contours on the frame for visualization"""
        if frame is None or eye_data is None:
            return frame
        
        # Clone the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Draw left eye
        hull = cv2.convexHull(eye_data['left_eye'])
        cv2.drawContours(vis_frame, [hull], -1, (0, 255, 0), 1)
        
        # Draw right eye
        hull = cv2.convexHull(eye_data['right_eye'])
        cv2.drawContours(vis_frame, [hull], -1, (0, 255, 0), 1)
        
        # Draw face rectangle
        x, y, w, h = (eye_data['face'].left(), 
                      eye_data['face'].top(),
                      eye_data['face'].width(),
                      eye_data['face'].height())
        cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return vis_frame
    
    def get_pupil_coordinates(self, eye_points):
        """
        Estimate pupil coordinates from eye contour using centroid
        This is a simplified approach; more accurate with MediaPipe's iris detection
        """
        if eye_points is None:
            return None
            
        # Use the centroid of the eye contour as an estimate
        centroid_x = np.mean([p[0] for p in eye_points])
        centroid_y = np.mean([p[1] for p in eye_points])
        
        return (int(centroid_x), int(centroid_y))
    
    def detect_blinks(self, eye_points, threshold=0.3):
        """
        Detect eye blinks using the eye aspect ratio (EAR)
        """
        if eye_points is None:
            return False
            
        # Calculate the euclidean distance between horizontal eye points
        width = np.linalg.norm(eye_points[0] - eye_points[8])
        
        # Calculate the average of two vertical distances
        height1 = np.linalg.norm(eye_points[4] - eye_points[12])
        height2 = np.linalg.norm(eye_points[5] - eye_points[11])
        height = (height1 + height2) / 2
        
        # Calculate eye aspect ratio
        ear = height / width
        
        # Return True if the eye is closed (EAR below threshold)
        return ear < threshold 