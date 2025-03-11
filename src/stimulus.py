import cv2
import numpy as np
import os
import time
from tkinter import Tk, Canvas, Label, Button, Frame
from PIL import Image, ImageTk
import threading

class StimulusPresenter:
    """
    Class for presenting stimulus images or interfaces for eye tracking
    Supports both OpenCV and Tkinter-based display methods
    """
    def __init__(self, stimulus_dir='static'):
        self.stimulus_dir = stimulus_dir
        self.current_stimulus = None
        self.display_window = None
        self.tk_root = None
        self.tk_canvas = None
        self.tk_image = None
        self.areas_of_interest = {}
        
        # Create stimulus directory if it doesn't exist
        if not os.path.exists(stimulus_dir):
            os.makedirs(stimulus_dir)
            
    def load_stimulus(self, stimulus_path):
        """Load a stimulus image"""
        if not os.path.exists(stimulus_path):
            raise FileNotFoundError(f"Stimulus file not found: {stimulus_path}")
            
        self.current_stimulus = stimulus_path
        return True
        
    def display_with_opencv(self, window_name='Stimulus', wait_key=False):
        """
        Display the current stimulus using OpenCV
        Note: This method blocks if wait_key is True
        """
        if self.current_stimulus is None:
            return False
            
        # Load and display the image
        img = cv2.imread(self.current_stimulus)
        if img is None:
            return False
            
        # Create or get the window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        self.display_window = window_name
        
        # Wait for key press if requested
        if wait_key:
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)
            self.display_window = None
            
        return True
        
    def close_opencv_window(self):
        """Close the OpenCV display window"""
        if self.display_window:
            cv2.destroyWindow(self.display_window)
            self.display_window = None
            
    def start_tkinter_display(self, width=1200, height=800, title="Eye Tracking Stimulus"):
        """
        Start a Tkinter-based display for the stimulus
        This is more flexible than OpenCV for complex interfaces
        """
        # Create Tkinter root if it doesn't exist
        if self.tk_root is None:
            self.tk_root = Tk()
            self.tk_root.title(title)
            self.tk_root.geometry(f"{width}x{height}")
            
            # Create a frame for the stimulus
            frame = Frame(self.tk_root, width=width, height=height)
            frame.pack(fill="both", expand=True)
            
            # Create canvas for the stimulus
            self.tk_canvas = Canvas(frame, width=width, height=height)
            self.tk_canvas.pack(fill="both", expand=True)
            
        return self.tk_root
        
    def update_tkinter_stimulus(self, stimulus_path=None):
        """Update the stimulus in the Tkinter window"""
        if stimulus_path:
            self.load_stimulus(stimulus_path)
            
        if self.current_stimulus is None or self.tk_canvas is None:
            return False
            
        # Load the image
        img = Image.open(self.current_stimulus)
        
        # Resize to fit canvas
        canvas_width = self.tk_canvas.winfo_width() or self.tk_root.winfo_width()
        canvas_height = self.tk_canvas.winfo_height() or self.tk_root.winfo_height()
        
        img = img.resize((canvas_width, canvas_height), Image.LANCZOS)
        
        # Convert to Tkinter-compatible photo image
        self.tk_image = ImageTk.PhotoImage(img)
        
        # Display on canvas
        self.tk_canvas.delete("all")  # Clear previous content
        self.tk_canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        
        return True
        
    def run_tkinter_loop(self):
        """Run the Tkinter main loop"""
        if self.tk_root:
            self.tk_root.mainloop()
            
    def close_tkinter_display(self):
        """Close the Tkinter display"""
        if self.tk_root:
            self.tk_root.destroy()
            self.tk_root = None
            self.tk_canvas = None
            self.tk_image = None
            
    def define_aoi(self, name, x, y, width, height):
        """Define an Area of Interest (AOI) for analysis"""
        self.areas_of_interest[name] = (x, y, width, height)
        return True
        
    def get_aois(self):
        """Get all defined Areas of Interest"""
        return self.areas_of_interest
        
    def highlight_aois(self, color="blue", outline_width=2):
        """
        Highlight defined AOIs on the Tkinter canvas
        Useful for debugging and demonstration
        """
        if not self.tk_canvas or not self.areas_of_interest:
            return False
            
        for name, (x, y, w, h) in self.areas_of_interest.items():
            # Draw rectangle
            self.tk_canvas.create_rectangle(
                x, y, x + w, y + h,
                outline=color, width=outline_width,
                tags="aoi"
            )
            
            # Add label
            self.tk_canvas.create_text(
                x + w//2, y - 10,
                text=name,
                fill=color,
                tags="aoi"
            )
            
        return True
        
    def clear_aoi_highlights(self):
        """Clear AOI highlights from the canvas"""
        if self.tk_canvas:
            self.tk_canvas.delete("aoi")
            return True
        return False
        
    def run_stimulus_sequence(self, stimulus_files, display_time=5):
        """
        Run a sequence of stimulus presentations with timing
        
        Parameters:
            stimulus_files (list): List of stimulus file paths
            display_time (float): Time in seconds to display each stimulus
            
        Returns:
            dict: Dictionary with stimulus files and their display times
        """
        results = {}
        
        # Start tkinter if not already running
        if not self.tk_root:
            self.start_tkinter_display()
            
        for stimulus_file in stimulus_files:
            # Load and display stimulus
            full_path = os.path.join(self.stimulus_dir, stimulus_file) if not os.path.isabs(stimulus_file) else stimulus_file
            self.update_tkinter_stimulus(full_path)
            
            # Record start time
            start_time = time.time()
            
            # Update the UI and wait
            self.tk_root.update()
            time.sleep(display_time)
            
            # Record duration
            end_time = time.time()
            results[stimulus_file] = end_time - start_time
            
        return results
        
    def interactive_display(self, callback=None):
        """
        Display stimulus with interactive elements
        
        Parameters:
            callback (function): Function to call when user interacts
                                with the stimulus
        """
        if not self.tk_root:
            self.start_tkinter_display()
            
        # Add a callback for mouse clicks if provided
        if callback:
            def on_click(event):
                callback(event.x, event.y)
                
            self.tk_canvas.bind("<Button-1>", on_click)
            
        # Run the tkinter loop
        self.run_tkinter_loop() 