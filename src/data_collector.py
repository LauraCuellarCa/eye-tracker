import csv
import time
import numpy as np
import pandas as pd
import os
from datetime import datetime

class DataCollector:
    """
    Class for collecting, recording, and storing gaze data
    """
    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        self.gaze_data = []
        self.recording = False
        self.start_time = None
        self.stimulus_name = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def start_recording(self, stimulus_name=None):
        """Start recording gaze data"""
        self.gaze_data = []
        self.recording = True
        self.start_time = time.time()
        self.stimulus_name = stimulus_name or f"stimulus_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return True
        
    def stop_recording(self):
        """Stop recording gaze data"""
        self.recording = False
        return len(self.gaze_data)
        
    def add_gaze_point(self, gaze_point, timestamp=None):
        """Add a gaze point to the data collection"""
        if not self.recording:
            return False
            
        if timestamp is None:
            timestamp = time.time() - self.start_time
            
        if gaze_point is not None:
            self.gaze_data.append({
                'timestamp': timestamp,
                'x': gaze_point[0],
                'y': gaze_point[1]
            })
            return True
        return False
        
    def save_data(self, filename=None):
        """Save collected gaze data to a CSV file"""
        if not self.gaze_data:
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.stimulus_name}_{timestamp}.csv"
            
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'x', 'y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data_point in self.gaze_data:
                writer.writerow(data_point)
                
        return filepath
        
    def get_data_as_dataframe(self):
        """Return recorded data as a pandas DataFrame"""
        if not self.gaze_data:
            return None
            
        return pd.DataFrame(self.gaze_data)
        
    def get_fixations(self, max_distance=50, min_duration=0.1):
        """
        Identify fixations in gaze data
        
        Parameters:
            max_distance (int): Maximum distance between points to be considered the same fixation
            min_duration (float): Minimum duration (in seconds) for a valid fixation
            
        Returns:
            list: List of fixation data points (x, y, duration)
        """
        if not self.gaze_data or len(self.gaze_data) < 3:
            return []
            
        # Convert to pandas DataFrame for easier manipulation
        df = self.get_data_as_dataframe()
        
        fixations = []
        current_fixation = []
        last_point = None
        
        for _, row in df.iterrows():
            point = (row['x'], row['y'])
            
            if last_point is None:
                # First point
                current_fixation = [point]
                last_point = point
                continue
                
            # Calculate distance to last point
            distance = np.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
            
            if distance <= max_distance:
                # Still in the same fixation
                current_fixation.append(point)
            else:
                # New fixation detected
                if len(current_fixation) > 0:
                    # Check if previous fixation meets minimum duration
                    fixation_df = df[df.index.isin(range(df.index[0], df.index[0] + len(current_fixation)))]
                    duration = fixation_df['timestamp'].max() - fixation_df['timestamp'].min()
                    
                    if duration >= min_duration:
                        # Calculate center of fixation
                        center_x = np.mean([p[0] for p in current_fixation])
                        center_y = np.mean([p[1] for p in current_fixation])
                        fixations.append({
                            'x': int(center_x),
                            'y': int(center_y),
                            'duration': duration,
                            'start_time': fixation_df['timestamp'].min(),
                            'end_time': fixation_df['timestamp'].max()
                        })
                        
                # Start new fixation
                current_fixation = [point]
                
            last_point = point
                
        # Check if the last fixation meets criteria
        if len(current_fixation) > 0:
            fixation_df = df.tail(len(current_fixation))
            duration = fixation_df['timestamp'].max() - fixation_df['timestamp'].min()
            
            if duration >= min_duration:
                center_x = np.mean([p[0] for p in current_fixation])
                center_y = np.mean([p[1] for p in current_fixation])
                fixations.append({
                    'x': int(center_x),
                    'y': int(center_y),
                    'duration': duration,
                    'start_time': fixation_df['timestamp'].min(),
                    'end_time': fixation_df['timestamp'].max()
                })
                
        return fixations 