import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

class ClickCollector:
    def __init__(self, image_path, max_size=(800, 600)):
        self.root = tk.Tk()
        self.root.title("Click Collector for SAM2")
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Use 80% of screen size as maximum if max_size not specified
        max_width = min(max_size[0], int(screen_width * 0.8))
        max_height = min(max_size[1], int(screen_height * 0.8))
        
        # Load and resize image
        self.original_image = Image.open(image_path)
        self.scale_factor = 1.0
        
        # Calculate scaling factor
        width, height = self.original_image.size
        if width > max_width or height > max_height:
            width_ratio = max_width / width
            height_ratio = max_height / height
            self.scale_factor = min(width_ratio, height_ratio)
            
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            self.image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            self.image = self.original_image
            
        self.photo = ImageTk.PhotoImage(self.image)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)
        
        # Create canvas for image
        self.canvas = tk.Canvas(
            self.main_frame, 
            width=self.image.width,
            height=self.image.height
        )
        self.canvas.pack()
        
        # Display image on canvas
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        
        # Initialize lists for current object's clicks
        self.current_clicks = []  # Will store (x, y) coordinates
        self.current_labels = []  # Will store 1 for positive (left click) and 0 for negative (right click)
        
        # Initialize lists for all objects
        self.all_points_list = []  # Will store arrays of points for each object
        self.all_labels_list = []  # Will store arrays of labels for each object
        
        # Store canvas items for current object
        self.current_markers = []
        
        # Bind mouse clicks
        self.canvas.bind("<Button-1>", lambda e: self.on_click(e, 1))  # Left click
        self.canvas.bind("<Button-3>", lambda e: self.on_click(e, 0))  # Right click
        
        # Create control buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(pady=10)
        
        ttk.Button(self.button_frame, text="Add Object", command=self.add_object).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Clear Current", command=self.clear_current).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Done", command=self.finish).pack(side=tk.LEFT, padx=5)
        
        # Object and click counter labels
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(pady=5)
        
        self.click_count = tk.StringVar(value="Current Object Clicks: 0")
        self.object_count = tk.StringVar(value="Total Objects: 0")
        ttk.Label(self.status_frame, textvariable=self.click_count).pack(side=tk.LEFT, padx=20)
        ttk.Label(self.status_frame, textvariable=self.object_count).pack(side=tk.LEFT, padx=20)
        
        # Instructions
        instructions = "Left click: Positive (green) | Right click: Negative (red)\n"
        instructions += "'Add Object': Save current clicks as object | 'Done': Finish all objects"
        ttk.Label(self.main_frame, text=instructions).pack(pady=5)
        
        self.finished = False
        
    def on_click(self, event, label):
        """Handle mouse clicks"""
        x, y = event.x, event.y
        
        # Store original-scale coordinates
        original_x = x / self.scale_factor
        original_y = y / self.scale_factor
        self.current_clicks.append([original_x, original_y])
        self.current_labels.append(label)
        
        # Draw marker on canvas
        color = 'green' if label == 1 else 'red'
        size = 5
        marker = self.canvas.create_oval(
            x - size, y - size, x + size, y + size, 
            fill=color, outline='white'
        )
        self.current_markers.append(marker)
        
        # Update click counter
        self.click_count.set(f"Current Object Clicks: {len(self.current_clicks)}")
        
    def add_object(self):
        """Add current clicks as a new object"""
        if self.current_clicks:  # Only add if there are clicks
            self.all_points_list.append(np.array(self.current_clicks, dtype=np.float32))
            self.all_labels_list.append(np.array(self.current_labels, dtype=np.int32))
            
            # Fade out current markers
            for marker in self.current_markers:
                self.canvas.itemconfig(marker, fill='gray', outline='gray')
            
            # Clear current clicks but keep markers visible
            self.current_clicks = []
            self.current_labels = []
            self.current_markers = []
            
            # Update counters
            self.click_count.set("Current Object Clicks: 0")
            self.object_count.set(f"Total Objects: {len(self.all_points_list)}")
        
    def clear_current(self):
        """Clear current object's clicks"""
        self.current_clicks = []
        self.current_labels = []
        # Delete current markers
        for marker in self.current_markers:
            self.canvas.delete(marker)
        self.current_markers = []
        self.click_count.set("Current Object Clicks: 0")
        
    def finish(self):
        """Complete the collection process"""
        # Add current object if there are clicks
        if self.current_clicks:
            self.add_object()
        self.finished = True
        self.root.quit()
        
    def get_results(self):
        """Return the collected clicks and labels as lists of numpy arrays"""
        if not self.all_points_list:
            # Return empty arrays with correct shape
            return [np.array([[]], dtype=np.float32).reshape((0, 2))], [np.array([], dtype=np.int32)]
        return self.all_points_list, self.all_labels_list
        
    def run(self):
        """Start the GUI"""
        # Center the window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')
        
        self.root.mainloop()
        self.root.destroy()
        return self.get_results()

def collect_clicks(image_path, max_size=(800, 600)):
    """
    Collect clicks for multiple objects from an image.
    
    Args:
        image_path (str): Path to the image file
        max_size (tuple): Maximum (width, height) for the displayed image
        
    Returns:
        tuple: (points_list, labels_list)
            - points_list: List of numpy arrays, each array of shape (n_clicks, 2)
            - labels_list: List of numpy arrays, each array of shape (n_clicks,)
    """
    collector = ClickCollector(image_path, max_size)
    return collector.run()