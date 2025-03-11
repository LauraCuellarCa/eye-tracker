# Eye Tracking Analysis System

A Python-based eye tracking solution for analyzing user attention on interfaces, products, or advertisements using webcam input.

## Features

- Real-time eye tracking using webcam
- Gaze estimation on screen coordinates
- Heatmap generation of attention areas
- Areas of Interest (AOI) analysis
- Data collection and visualization
- Simple stimuli presentation

## Prerequisites

- Python 3.8 or newer
- Webcam
- Required packages (see requirements.txt)

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   
3. Run the application:
   ```
   python run_eye_tracker.py
   ```

## Project Structure

- `src/`: Source code
  - `eye_detector.py`: Eye detection from webcam using MediaPipe
  - `gaze_estimator.py`: Gaze point estimation
  - `data_collector.py`: Data recording functions
  - `visualizer.py`: Heatmaps and visualization
  - `stimulus.py`: Stimulus presentation
  - `main.py`: Main application
- `data/`: Raw gaze data storage
- `results/`: Generated heatmaps and analysis results
- `static/`: Stimulus images and test materials

## Usage

1. Run the application
2. Follow calibration instructions
3. Load stimulus materials
4. Begin recording eye tracking data
5. Generate visualizations and analysis

## Technology Stack

This application uses MediaPipe for facial landmark detection, which provides:
- Fast and accurate face mesh detection
- 468 facial landmarks (including detailed eye regions)
- Iris detection capabilities
- No need for additional model files

## Troubleshooting

- **Webcam access**: Ensure your webcam is properly connected and not being used by another application.
- **Import errors**: Make sure your Python path is set correctly and all dependencies are installed.