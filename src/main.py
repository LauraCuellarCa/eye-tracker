import os
import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

# Import our modules
from src.eye_detector import EyeDetector
from src.gaze_estimator import GazeEstimator
from src.data_collector import DataCollector
from src.visualizer import Visualizer
from src.stimulus import StimulusPresenter

class EyeTrackingApp:
    """
    Main application class that integrates all the eye tracking components
    """
    def __init__(self, root=None):
        # Initialize components
        self.eye_detector = EyeDetector()
        self.gaze_estimator = GazeEstimator()
        self.data_collector = DataCollector()
        self.visualizer = Visualizer()
        self.stimulus_presenter = StimulusPresenter()
        
        # Setup UI
        self.root = root
        if not self.root:
            self.root = tk.Tk()
            self.root.title("Eye Tracking Analysis System")
            self.root.geometry("1600x900")
            
        # Create main frames
        self.setup_ui()
        
        # Initialize variables
        self.webcam_active = False
        self.recording = False
        self.calibration_mode = False
        self.calibration_points = []
        self.current_calibration_point = None
        self.stop_event = threading.Event()
        self.webcam_thread = None
        self.fixation_threshold = 50  # pixels
        self.fixation_time = 0.1  # seconds
        self.current_stimulus = None
        self.defined_aois = {}
        
    def setup_ui(self):
        """Set up the application UI"""
        # Create main frames
        self.left_frame = ttk.Frame(self.root, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.root, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Video display area
        self.video_label = ttk.Label(self.left_frame)
        self.video_label.pack(pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Start webcam to begin.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Control panel
        control_frame = ttk.LabelFrame(self.right_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Webcam control
        self.webcam_btn = ttk.Button(control_frame, text="Start Webcam", 
                                     command=self.toggle_webcam)
        self.webcam_btn.pack(fill=tk.X, pady=5)
        
        # Calibration button
        self.calibration_btn = ttk.Button(control_frame, text="Start Calibration", 
                                         command=self.start_calibration, state=tk.DISABLED)
        self.calibration_btn.pack(fill=tk.X, pady=5)
        
        # Recording button
        self.recording_btn = ttk.Button(control_frame, text="Start Recording", 
                                       command=self.toggle_recording, state=tk.DISABLED)
        self.recording_btn.pack(fill=tk.X, pady=5)
        
        # Stimulus section
        stimulus_frame = ttk.LabelFrame(self.right_frame, text="Stimulus", padding=10)
        stimulus_frame.pack(fill=tk.X, pady=5)
        
        # Load stimulus button
        self.load_stimulus_btn = ttk.Button(stimulus_frame, text="Load Stimulus Image", 
                                           command=self.load_stimulus)
        self.load_stimulus_btn.pack(fill=tk.X, pady=5)
        
        # Display stimulus button
        self.display_stimulus_btn = ttk.Button(stimulus_frame, text="Display Stimulus", 
                                             command=self.display_stimulus, state=tk.DISABLED)
        self.display_stimulus_btn.pack(fill=tk.X, pady=5)
        
        # AOI button
        self.aoi_btn = ttk.Button(stimulus_frame, text="Define Areas of Interest", 
                                 command=self.define_aois, state=tk.DISABLED)
        self.aoi_btn.pack(fill=tk.X, pady=5)
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(self.right_frame, text="Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, pady=5)
        
        # Generate Heatmap button
        self.heatmap_btn = ttk.Button(analysis_frame, text="Generate Heatmap", 
                                     command=self.generate_heatmap, state=tk.DISABLED)
        self.heatmap_btn.pack(fill=tk.X, pady=5)
        
        # Generate Scanpath button
        self.scanpath_btn = ttk.Button(analysis_frame, text="Generate Scanpath", 
                                      command=self.generate_scanpath, state=tk.DISABLED)
        self.scanpath_btn.pack(fill=tk.X, pady=5)
        
        # AOI Analysis button
        self.aoi_analysis_btn = ttk.Button(analysis_frame, text="Analyze AOIs", 
                                         command=self.analyze_aois, state=tk.DISABLED)
        self.aoi_analysis_btn.pack(fill=tk.X, pady=5)
        
        # Settings section
        settings_frame = ttk.LabelFrame(self.right_frame, text="Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Fixation threshold
        ttk.Label(settings_frame, text="Fixation Distance Threshold (px):").pack(anchor=tk.W)
        self.fixation_threshold_var = tk.IntVar(value=50)
        threshold_slider = ttk.Scale(settings_frame, from_=10, to=100, 
                                    variable=self.fixation_threshold_var, 
                                    orient=tk.HORIZONTAL)
        threshold_slider.pack(fill=tk.X)
        
        # Fixation time
        ttk.Label(settings_frame, text="Minimum Fixation Time (s):").pack(anchor=tk.W)
        self.fixation_time_var = tk.DoubleVar(value=0.1)
        time_slider = ttk.Scale(settings_frame, from_=0.05, to=0.5, 
                              variable=self.fixation_time_var, 
                              orient=tk.HORIZONTAL)
        time_slider.pack(fill=tk.X)
        
        # Exit button
        exit_btn = ttk.Button(self.right_frame, text="Exit", command=self.on_exit)
        exit_btn.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
    def toggle_webcam(self):
        """Start or stop the webcam"""
        if not self.webcam_active:
            # Start webcam
            try:
                if self.eye_detector.start_webcam():
                    self.webcam_active = True
                    self.webcam_btn.config(text="Stop Webcam")
                    self.calibration_btn.config(state=tk.NORMAL)
                    self.status_var.set("Webcam started. Ready for calibration.")
                    
                    # Start webcam thread
                    self.stop_event.clear()
                    self.webcam_thread = threading.Thread(target=self.webcam_loop)
                    self.webcam_thread.daemon = True
                    self.webcam_thread.start()
                else:
                    messagebox.showerror("Error", "Could not start webcam")
            except Exception as e:
                messagebox.showerror("Error", f"Error starting webcam: {str(e)}")
        else:
            # Stop webcam
            self.stop_event.set()
            if self.webcam_thread:
                self.webcam_thread.join(timeout=1.0)
            
            self.eye_detector.stop_webcam()
            self.webcam_active = False
            self.webcam_btn.config(text="Start Webcam")
            self.calibration_btn.config(state=tk.DISABLED)
            self.recording_btn.config(state=tk.DISABLED)
            self.status_var.set("Webcam stopped")
            
            # Clear video display
            self.video_label.config(image='')
            
    def webcam_loop(self):
        """Main webcam processing loop that runs in a separate thread"""
        while not self.stop_event.is_set():
            # Get frame from webcam
            frame = self.eye_detector.get_frame()
            if frame is None:
                continue
                
            # Detect eyes
            eye_data = self.eye_detector.detect_eyes(frame)
            
            # Draw eyes on the frame for visualization
            if eye_data:
                frame = self.eye_detector.draw_eyes(frame, eye_data)
                
                # If calibrated, estimate gaze point
                if self.gaze_estimator.is_calibrated and not self.calibration_mode:
                    gaze_point = self.gaze_estimator.estimate_gaze_point(eye_data)
                    
                    if gaze_point:
                        # Draw gaze point on frame
                        cv2.circle(frame, gaze_point, 10, (0, 0, 255), -1)
                        
                        # If recording, add to data collector
                        if self.recording:
                            self.data_collector.add_gaze_point(gaze_point)
                
                # If in calibration mode, handle calibration
                if self.calibration_mode and self.current_calibration_point:
                    # Extract eye features
                    eye_features = self.gaze_estimator.extract_eye_features(eye_data)
                    
                    if eye_features is not None:
                        # Draw current calibration target
                        target_x, target_y = self.current_calibration_point
                        cv2.circle(frame, (target_x, target_y), 15, (255, 0, 0), -1)
                        cv2.circle(frame, (target_x, target_y), 5, (255, 255, 255), -1)
            
            # Convert frame to format suitable for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update video display in the main thread
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            
            # Short sleep to reduce CPU usage
            time.sleep(0.01)
            
    def start_calibration(self):
        """Start the calibration process"""
        if not self.webcam_active:
            messagebox.showerror("Error", "Webcam must be active to calibrate")
            return
            
        self.calibration_mode = True
        self.calibration_points = []
        self.status_var.set("Calibration started. Follow the circles.")
        
        # Disable buttons during calibration
        self.calibration_btn.config(state=tk.DISABLED)
        self.recording_btn.config(state=tk.DISABLED)
        
        # Run calibration in a separate thread
        threading.Thread(target=self.calibration_sequence).start()
        
    def calibration_sequence(self):
        """Run the calibration sequence with points at different screen positions"""
        # Get video label dimensions
        width = self.video_label.winfo_width()
        height = self.video_label.winfo_height()
        
        # Define calibration points (9-point calibration)
        points = [
            (width // 4, height // 4),           # Top-left
            (width // 2, height // 4),           # Top-center
            (3 * width // 4, height // 4),       # Top-right
            (width // 4, height // 2),           # Mid-left
            (width // 2, height // 2),           # Center
            (3 * width // 4, height // 2),       # Mid-right
            (width // 4, 3 * height // 4),       # Bottom-left
            (width // 2, 3 * height // 4),       # Bottom-center
            (3 * width // 4, 3 * height // 4)    # Bottom-right
        ]
        
        # For each point, collect calibration data
        for point in points:
            self.current_calibration_point = point
            
            # Wait for a moment to let user focus on the point
            time.sleep(2)
            
            # Get current eye data
            frame = self.eye_detector.get_frame()
            if frame is None:
                continue
                
            eye_data = self.eye_detector.detect_eyes(frame)
            if eye_data is None:
                continue
                
            # Extract features and add to calibration
            eye_features = self.gaze_estimator.extract_eye_features(eye_data)
            if eye_features is not None:
                # Scale point to screen coordinates
                screen_x = int(point[0] * (self.gaze_estimator.screen_width / width))
                screen_y = int(point[1] * (self.gaze_estimator.screen_height / height))
                
                # Add to calibration data
                self.gaze_estimator.add_calibration_point(
                    (screen_x, screen_y), eye_features)
                
                self.calibration_points.append(point)
                
        # Finish calibration
        success = self.gaze_estimator.calibrate()
        
        # Reset calibration mode
        self.calibration_mode = False
        self.current_calibration_point = None
        
        # Update UI
        if success:
            self.status_var.set("Calibration successful. Ready to record.")
            self.recording_btn.config(state=tk.NORMAL)
        else:
            self.status_var.set("Calibration failed. Try again.")
            
        self.calibration_btn.config(state=tk.NORMAL)
        
    def toggle_recording(self):
        """Start or stop recording eye tracking data"""
        if not self.recording:
            # Start recording
            stimulus_name = None
            if self.current_stimulus:
                stimulus_name = os.path.splitext(os.path.basename(self.current_stimulus))[0]
                
            self.data_collector.start_recording(stimulus_name)
            self.recording = True
            self.recording_btn.config(text="Stop Recording")
            self.status_var.set("Recording data...")
            
            # Disable certain buttons during recording
            self.calibration_btn.config(state=tk.DISABLED)
        else:
            # Stop recording
            count = self.data_collector.stop_recording()
            self.recording = False
            self.recording_btn.config(text="Start Recording")
            
            # Save data
            filepath = self.data_collector.save_data()
            
            self.status_var.set(f"Recording stopped. {count} gaze points saved to {filepath}")
            
            # Enable analysis buttons if we collected data
            if count > 0:
                self.heatmap_btn.config(state=tk.NORMAL)
                self.scanpath_btn.config(state=tk.NORMAL)
                if self.defined_aois:
                    self.aoi_analysis_btn.config(state=tk.NORMAL)
                    
            # Re-enable buttons
            self.calibration_btn.config(state=tk.NORMAL)
            
    def load_stimulus(self):
        """Load a stimulus image"""
        file_path = filedialog.askopenfilename(
            title="Select Stimulus Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            try:
                self.stimulus_presenter.load_stimulus(file_path)
                self.current_stimulus = file_path
                self.display_stimulus_btn.config(state=tk.NORMAL)
                self.aoi_btn.config(state=tk.NORMAL)
                self.status_var.set(f"Stimulus loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load stimulus: {str(e)}")
                
    def display_stimulus(self):
        """Display the loaded stimulus image"""
        if not self.current_stimulus:
            messagebox.showerror("Error", "No stimulus loaded")
            return
            
        # Create a new window for stimulus display
        stimulus_window = tk.Toplevel(self.root)
        stimulus_window.title("Stimulus Display")
        
        # Display the image
        img = Image.open(self.current_stimulus)
        img = img.resize((800, 600), Image.LANCZOS)  # Resize for display
        photo = ImageTk.PhotoImage(img)
        
        # Create label to display image
        label = ttk.Label(stimulus_window, image=photo)
        label.image = photo  # Keep a reference
        label.pack(padx=10, pady=10)
        
        # Display AOIs if defined
        if self.defined_aois:
            canvas = tk.Canvas(stimulus_window, width=800, height=600)
            canvas.pack()
            
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            
            # Draw AOIs
            for name, (x, y, w, h) in self.defined_aois.items():
                # Scale to fit 800x600 display
                scaled_x = int(x * 800 / img.width)
                scaled_y = int(y * 600 / img.height)
                scaled_w = int(w * 800 / img.width)
                scaled_h = int(h * 600 / img.height)
                
                canvas.create_rectangle(
                    scaled_x, scaled_y, 
                    scaled_x + scaled_w, scaled_y + scaled_h,
                    outline="blue", width=2
                )
                
                canvas.create_text(
                    scaled_x + scaled_w//2, scaled_y - 10,
                    text=name, fill="blue"
                )
        
    def define_aois(self):
        """Open a window to define Areas of Interest on the stimulus"""
        if not self.current_stimulus:
            messagebox.showerror("Error", "No stimulus loaded")
            return
            
        # Create a new window for AOI definition
        aoi_window = tk.Toplevel(self.root)
        aoi_window.title("Define Areas of Interest")
        
        # Display the image
        img = Image.open(self.current_stimulus)
        img = img.resize((800, 600), Image.LANCZOS)  # Resize for display
        photo = ImageTk.PhotoImage(img)
        
        # Create canvas for interactive AOI definition
        canvas = tk.Canvas(aoi_window, width=800, height=600)
        canvas.pack(side=tk.LEFT)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo  # Keep a reference
        
        # AOI list and controls
        control_frame = ttk.Frame(aoi_window, padding=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Label(control_frame, text="Areas of Interest").pack(anchor=tk.W)
        
        # List of defined AOIs
        aoi_listbox = tk.Listbox(control_frame, width=30, height=15)
        aoi_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Update listbox with existing AOIs
        for name in self.defined_aois:
            aoi_listbox.insert(tk.END, name)
            
        # AOI name entry
        ttk.Label(control_frame, text="AOI Name:").pack(anchor=tk.W)
        name_var = tk.StringVar()
        name_entry = ttk.Entry(control_frame, textvariable=name_var)
        name_entry.pack(fill=tk.X, pady=5)
        
        # Instructions
        ttk.Label(control_frame, 
                 text="Click and drag to define an AOI.\n"
                      "Enter a name and click Add.").pack(pady=10)
        
        # Variables to track rectangle drawing
        start_x, start_y = 0, 0
        rect_id = None
        
        # Function to handle mouse events
        def on_mouse_down(event):
            nonlocal start_x, start_y, rect_id
            start_x, start_y = event.x, event.y
            rect_id = canvas.create_rectangle(
                start_x, start_y, start_x, start_y,
                outline="red", width=2, tags="temp_rect"
            )
            
        def on_mouse_move(event):
            nonlocal rect_id
            if rect_id:
                canvas.coords(rect_id, start_x, start_y, event.x, event.y)
                
        def on_mouse_up(event):
            # Keep the rectangle until explicitly cleared
            pass
            
        # Attach mouse events
        canvas.bind("<ButtonPress-1>", on_mouse_down)
        canvas.bind("<B1-Motion>", on_mouse_move)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)
        
        # Function to add the current AOI
        def add_aoi():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter an AOI name")
                return
                
            # Get coordinates of the drawn rectangle
            if rect_id:
                coords = canvas.coords(rect_id)
                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    
                    # Ensure x1,y1 is the top-left and x2,y2 is the bottom-right
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                        
                    # Calculate width and height
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Scale back to original image size
                    orig_x = int(x1 * img.width / 800)
                    orig_y = int(y1 * img.height / 600)
                    orig_w = int(w * img.width / 800)
                    orig_h = int(h * img.height / 600)
                    
                    # Add to AOIs
                    self.defined_aois[name] = (orig_x, orig_y, orig_w, orig_h)
                    
                    # Add to listbox
                    aoi_listbox.insert(tk.END, name)
                    
                    # Clear the temporary rectangle
                    canvas.delete("temp_rect")
                    rect_id = None
                    
                    # Clear the name field
                    name_var.set("")
                    
                    # Draw the AOI with its name
                    canvas.create_rectangle(
                        x1, y1, x2, y2,
                        outline="blue", width=2, tags="aoi"
                    )
                    
                    canvas.create_text(
                        (x1 + x2) / 2, y1 - 5,
                        text=name, fill="blue", tags="aoi"
                    )
                    
                    # Enable AOI analysis button
                    self.aoi_analysis_btn.config(state=tk.NORMAL)
            
        # Function to remove selected AOI
        def remove_aoi():
            selected = aoi_listbox.curselection()
            if selected:
                name = aoi_listbox.get(selected[0])
                if name in self.defined_aois:
                    del self.defined_aois[name]
                    aoi_listbox.delete(selected[0])
                    
                    # Redraw all AOIs
                    canvas.delete("aoi")
                    for aoi_name, (x, y, w, h) in self.defined_aois.items():
                        # Scale for display
                        scaled_x = int(x * 800 / img.width)
                        scaled_y = int(y * 600 / img.height)
                        scaled_w = int(w * 800 / img.width)
                        scaled_h = int(h * 600 / img.height)
                        
                        canvas.create_rectangle(
                            scaled_x, scaled_y, 
                            scaled_x + scaled_w, scaled_y + scaled_h,
                            outline="blue", width=2, tags="aoi"
                        )
                        
                        canvas.create_text(
                            scaled_x + scaled_w//2, scaled_y - 5,
                            text=aoi_name, fill="blue", tags="aoi"
                        )
                        
                    # Disable AOI analysis button if no AOIs left
                    if not self.defined_aois:
                        self.aoi_analysis_btn.config(state=tk.DISABLED)
        
        # Function to clear all drawings
        def clear_canvas():
            canvas.delete("temp_rect")
            nonlocal rect_id
            rect_id = None
            
        # Add buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        add_btn = ttk.Button(button_frame, text="Add AOI", command=add_aoi)
        add_btn.pack(side=tk.LEFT, padx=5)
        
        remove_btn = ttk.Button(button_frame, text="Remove", command=remove_aoi)
        remove_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(button_frame, text="Clear", command=clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = ttk.Button(control_frame, text="Close", 
                              command=aoi_window.destroy)
        close_btn.pack(fill=tk.X, pady=10)
        
    def generate_heatmap(self):
        """Generate a heatmap visualization from the recorded data"""
        # Get fixations from recorded data
        if not self.data_collector.gaze_data:
            messagebox.showerror("Error", "No eye tracking data recorded")
            return
            
        # Get parameters from settings
        threshold = self.fixation_threshold_var.get()
        min_time = self.fixation_time_var.get()
        
        # Get fixations
        fixations = self.data_collector.get_fixations(threshold, min_time)
        
        if not fixations:
            messagebox.showerror("Error", "No fixations detected in the data")
            return
            
        # Generate heatmap with the stimulus as background (if available)
        output_path = self.visualizer.generate_heatmap(
            fixations, 
            image_path=self.current_stimulus
        )
        
        if output_path:
            self.status_var.set(f"Heatmap saved to {output_path}")
            
            # Open the heatmap
            try:
                if sys.platform == 'win32':
                    os.startfile(output_path)
                elif sys.platform == 'darwin':  # macOS
                    import subprocess
                    subprocess.call(['open', output_path])
                else:  # Linux
                    import subprocess
                    subprocess.call(['xdg-open', output_path])
            except Exception:
                pass
        else:
            messagebox.showerror("Error", "Failed to generate heatmap")
            
    def generate_scanpath(self):
        """Generate a scanpath visualization from the recorded data"""
        # Similar to generate_heatmap but for scanpath
        if not self.data_collector.gaze_data:
            messagebox.showerror("Error", "No eye tracking data recorded")
            return
            
        # Get parameters from settings
        threshold = self.fixation_threshold_var.get()
        min_time = self.fixation_time_var.get()
        
        # Get fixations
        fixations = self.data_collector.get_fixations(threshold, min_time)
        
        if not fixations:
            messagebox.showerror("Error", "No fixations detected in the data")
            return
            
        # Generate scanpath with the stimulus as background (if available)
        output_path = self.visualizer.generate_scanpath(
            fixations, 
            image_path=self.current_stimulus
        )
        
        if output_path:
            self.status_var.set(f"Scanpath saved to {output_path}")
            
            # Open the scanpath
            try:
                if sys.platform == 'win32':
                    os.startfile(output_path)
                elif sys.platform == 'darwin':  # macOS
                    import subprocess
                    subprocess.call(['open', output_path])
                else:  # Linux
                    import subprocess
                    subprocess.call(['xdg-open', output_path])
            except Exception:
                pass
        else:
            messagebox.showerror("Error", "Failed to generate scanpath")
            
    def analyze_aois(self):
        """Analyze defined Areas of Interest in relation to fixation data"""
        if not self.data_collector.gaze_data:
            messagebox.showerror("Error", "No eye tracking data recorded")
            return
            
        if not self.defined_aois:
            messagebox.showerror("Error", "No Areas of Interest defined")
            return
            
        # Get parameters from settings
        threshold = self.fixation_threshold_var.get()
        min_time = self.fixation_time_var.get()
        
        # Get fixations
        fixations = self.data_collector.get_fixations(threshold, min_time)
        
        if not fixations:
            messagebox.showerror("Error", "No fixations detected in the data")
            return
            
        # Analyze AOIs
        output_path, metrics = self.visualizer.analyze_aois(
            fixations, 
            self.defined_aois,
            image_path=self.current_stimulus
        )
        
        if output_path and metrics:
            self.status_var.set(f"AOI analysis saved to {output_path}")
            
            # Show metrics in a separate window
            metrics_window = tk.Toplevel(self.root)
            metrics_window.title("AOI Analysis Results")
            
            # Create a table of metrics
            table_frame = ttk.Frame(metrics_window, padding=10)
            table_frame.pack(fill=tk.BOTH, expand=True)
            
            # Table headers
            headers = ["AOI", "Fixation Count", "Duration (s)", 
                      "Time to First Fixation (s)", "Percentage"]
            
            for col, header in enumerate(headers):
                ttk.Label(table_frame, text=header, font=("Arial", 10, "bold")).grid(
                    row=0, column=col, padx=5, pady=5, sticky=tk.W)
                
            # Table data
            for row, (aoi_name, aoi_metrics) in enumerate(metrics.items(), 1):
                ttk.Label(table_frame, text=aoi_name).grid(
                    row=row, column=0, padx=5, pady=2, sticky=tk.W)
                    
                ttk.Label(table_frame, text=str(aoi_metrics['fixation_count'])).grid(
                    row=row, column=1, padx=5, pady=2)
                    
                ttk.Label(table_frame, text=f"{aoi_metrics['total_duration']:.2f}").grid(
                    row=row, column=2, padx=5, pady=2)
                    
                ttf = aoi_metrics['time_to_first_fixation']
                ttf_text = f"{ttf:.2f}" if ttf is not None else "N/A"
                ttk.Label(table_frame, text=ttf_text).grid(
                    row=row, column=3, padx=5, pady=2)
                    
                percentage = aoi_metrics['percentage_of_total'] * 100
                ttk.Label(table_frame, text=f"{percentage:.1f}%").grid(
                    row=row, column=4, padx=5, pady=2)
                    
            # Open the visualization
            try:
                if sys.platform == 'win32':
                    os.startfile(output_path)
                elif sys.platform == 'darwin':  # macOS
                    import subprocess
                    subprocess.call(['open', output_path])
                else:  # Linux
                    import subprocess
                    subprocess.call(['xdg-open', output_path])
            except Exception:
                pass
        else:
            messagebox.showerror("Error", "Failed to analyze AOIs")
            
    def on_exit(self):
        """Clean up resources and exit"""
        # Stop webcam if active
        if self.webcam_active:
            self.stop_event.set()
            if self.webcam_thread:
                self.webcam_thread.join(timeout=1.0)
            self.eye_detector.stop_webcam()
            
        # Destroy main window
        self.root.destroy()
        
def main():
    """Main entry point for the application"""
    root = tk.Tk()
    app = EyeTrackingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()
    
if __name__ == "__main__":
    main() 