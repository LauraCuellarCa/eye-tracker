import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cv2
from PIL import Image
import os

class Visualizer:
    """
    Class for visualizing eye tracking data through heatmaps, scanpaths, and AOI analysis
    """
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate_heatmap(self, fixations, image_path=None, output_path=None, 
                         alpha=0.7, gaussian_sigma=50, width=1920, height=1080):
        """
        Generate a heatmap visualization from fixation data
        
        Parameters:
            fixations (list): List of fixation dictionaries with 'x', 'y', 'duration'
            image_path (str): Path to background image (optional)
            output_path (str): Path to save visualization
            alpha (float): Transparency of heatmap overlay
            gaussian_sigma (int): Size of gaussian blur for heatmap
            width (int): Width of output image
            height (int): Height of output image
            
        Returns:
            str: Path to saved visualization
        """
        if not fixations:
            return None
        
        # Create a blank heat map
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Accumulate fixation durations at each point
        for fixation in fixations:
            x, y = int(fixation['x']), int(fixation['y'])
            
            # Ensure coordinates are within bounds
            if 0 <= x < width and 0 <= y < height:
                # Weight by duration
                weight = fixation['duration'] * 10  # Scale factor for visibility
                heatmap[y, x] += weight
        
        # Apply Gaussian blur to create heatmap effect
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), gaussian_sigma)
        
        # Normalize heatmap values to 0-1 range
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Create colormap for heatmap
        colormap = plt.get_cmap('jet')
        heatmap_colored = colormap(heatmap)
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # If background image is provided, overlay heatmap
        if image_path and os.path.exists(image_path):
            # Load background image
            bg_img = cv2.imread(image_path)
            
            # Resize if dimensions don't match
            if bg_img.shape[1] != width or bg_img.shape[0] != height:
                bg_img = cv2.resize(bg_img, (width, height))
                
            # Convert BGR to RGB
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            
            # Create mask based on heatmap intensity
            mask = np.zeros((height, width), dtype=np.uint8)
            normalized = Normalize()(heatmap)
            mask = (normalized * 255).astype(np.uint8)
            
            # Create final overlay image
            heatmap_overlay = np.zeros((height, width, 4), dtype=np.uint8)
            for c in range(3):  # RGB channels
                heatmap_overlay[:, :, c] = heatmap_colored[:, :, c]
            heatmap_overlay[:, :, 3] = (mask * alpha).astype(np.uint8)  # Alpha channel
            
            # Convert to PIL images for alpha composition
            background = Image.fromarray(bg_img)
            overlay = Image.fromarray(heatmap_overlay)
            
            # Composite with alpha
            background.paste(overlay, (0, 0), overlay)
            
            # Convert back to numpy array
            result = np.array(background)
        else:
            # No background image, just use colored heatmap
            result = heatmap_colored[:, :, :3]
        
        # Determine output path
        if output_path is None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0] if image_path else 'heatmap'
            output_path = os.path.join(self.output_dir, f"{base_name}_heatmap.png")
        
        # Save result
        plt.figure(figsize=(12, 8))
        plt.imshow(result)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_scanpath(self, fixations, image_path=None, output_path=None,
                         width=1920, height=1080, show_durations=True):
        """
        Generate a scanpath visualization from fixation data
        
        Parameters:
            fixations (list): List of fixation dictionaries
            image_path (str): Path to background image (optional)
            output_path (str): Path to save visualization
            width (int): Width of output image
            height (int): Height of output image
            show_durations (bool): Whether to show fixation durations
            
        Returns:
            str: Path to saved visualization
        """
        if not fixations:
            return None
            
        plt.figure(figsize=(12, 8))
        
        # If background image is provided, display it
        if image_path and os.path.exists(image_path):
            img = plt.imread(image_path)
            plt.imshow(img, extent=[0, width, height, 0])
        else:
            plt.xlim(0, width)
            plt.ylim(height, 0)  # Invert y-axis to match screen coordinates
        
        # Plot fixation points and connecting lines
        xs = [f['x'] for f in fixations]
        ys = [f['y'] for f in fixations]
        durations = [f['duration'] for f in fixations]
        
        # Scale point sizes based on duration
        sizes = [max(50, d * 100) for d in durations] if show_durations else [100] * len(fixations)
        
        # Plot scanpath lines
        plt.plot(xs, ys, 'b-', alpha=0.5, linewidth=1)
        
        # Plot fixation points
        scatter = plt.scatter(xs, ys, s=sizes, c=range(len(xs)), 
                             cmap='viridis', alpha=0.6, edgecolors='white')
        
        # Add numbers to indicate sequence
        for i, (x, y) in enumerate(zip(xs, ys)):
            plt.text(x, y, str(i+1), color='white', fontsize=8, 
                     ha='center', va='center')
        
        plt.colorbar(scatter, label='Fixation sequence')
        plt.title('Scanpath Visualization')
        plt.axis('off')
        
        # Determine output path
        if output_path is None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0] if image_path else 'scanpath'
            output_path = os.path.join(self.output_dir, f"{base_name}_scanpath.png")
        
        # Save result
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def analyze_aois(self, fixations, aois, image_path=None, output_path=None, 
                    width=1920, height=1080):
        """
        Analyze and visualize Areas of Interest (AOIs)
        
        Parameters:
            fixations (list): List of fixation dictionaries
            aois (dict): Dictionary of AOIs with name and bounding box (x, y, w, h)
            image_path (str): Path to background image (optional)
            output_path (str): Path to save visualization
            width (int): Width of image
            height (int): Height of image
            
        Returns:
            tuple: (visualization_path, aoi_metrics)
        """
        if not fixations or not aois:
            return None, None
            
        # Calculate metrics for each AOI
        aoi_metrics = {}
        for aoi_name, aoi_box in aois.items():
            x, y, w, h = aoi_box
            
            # Find fixations within this AOI
            aoi_fixations = []
            for fixation in fixations:
                fx, fy = fixation['x'], fixation['y']
                if x <= fx <= x + w and y <= fy <= y + h:
                    aoi_fixations.append(fixation)
            
            # Calculate metrics
            total_duration = sum(f['duration'] for f in aoi_fixations)
            total_count = len(aoi_fixations)
            
            # Find first fixation in this AOI (if any)
            first_fixation_time = None
            if aoi_fixations:
                first_fixation_time = min(f['start_time'] for f in aoi_fixations)
            
            aoi_metrics[aoi_name] = {
                'total_duration': total_duration,
                'fixation_count': total_count,
                'time_to_first_fixation': first_fixation_time,
                'percentage_of_total': total_duration / sum(f['duration'] for f in fixations) if fixations else 0
            }
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # If background image is provided, display it
        if image_path and os.path.exists(image_path):
            img = plt.imread(image_path)
            plt.imshow(img, extent=[0, width, height, 0])
        else:
            plt.xlim(0, width)
            plt.ylim(height, 0)  # Invert y-axis to match screen coordinates
        
        # Draw AOIs
        for aoi_name, aoi_box in aois.items():
            x, y, w, h = aoi_box
            metrics = aoi_metrics[aoi_name]
            
            # Color based on fixation duration percentage
            color = plt.cm.viridis(metrics['percentage_of_total'])
            
            # Draw rectangle
            rect = plt.Rectangle((x, y), w, h, linewidth=2, 
                                 edgecolor=color, facecolor='none', alpha=0.8)
            plt.gca().add_patch(rect)
            
            # Add label with metrics
            label = f"{aoi_name}\n{metrics['fixation_count']} fixations\n{metrics['total_duration']:.2f}s"
            plt.text(x + w/2, y + h/2, label, 
                     ha='center', va='center', fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.7))
        
        # Plot fixation points
        xs = [f['x'] for f in fixations]
        ys = [f['y'] for f in fixations]
        durations = [f['duration'] for f in fixations]
        
        plt.scatter(xs, ys, s=[d * 100 for d in durations], alpha=0.5, 
                   c='red', edgecolors='white')
        
        plt.title('Areas of Interest Analysis')
        plt.axis('off')
        
        # Determine output path
        if output_path is None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0] if image_path else 'aoi_analysis'
            output_path = os.path.join(self.output_dir, f"{base_name}_aoi_analysis.png")
        
        # Save result
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path, aoi_metrics 