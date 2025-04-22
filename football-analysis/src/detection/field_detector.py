import numpy as np

# Add CV2 stubs for IDE to recognize methods
try:
    import cv2
    # Test if methods exist
    cv2.cvtColor
    cv2.findContours
    cv2.convexHull
    cv2.bitwise_or
    cv2.threshold
    cv2.THRESH_BINARY
    cv2.Canny
    cv2.HoughLinesP
except (ImportError, AttributeError):
    print("Warning: Some cv2 methods not found, using stub implementation")
    # Create comprehensive stub implementation for IDE
    class CV2Stub:
        # Constants for color conversion
        COLOR_BGR2HSV = 40
        COLOR_BGR2GRAY = 6
        COLOR_BGR2RGB = 4
        COLOR_RGB2BGR = 4
        
        # Constants for thresholding
        THRESH_BINARY = 0
        THRESH_BINARY_INV = 1
        THRESH_TRUNC = 2
        THRESH_TOZERO = 3
        THRESH_TOZERO_INV = 4
        
        # Constants for morphology
        MORPH_OPEN = 2
        MORPH_CLOSE = 3
        MORPH_DILATE = 1
        MORPH_ERODE = 0
        
        # Constants for contour retrieval
        RETR_EXTERNAL = 0
        CHAIN_APPROX_SIMPLE = 1
        FONT_HERSHEY_SIMPLEX = 0
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
        
        def cvtColor(self, src, code, dstCn=None):
            """Stub for cv2.cvtColor"""
            if code == self.COLOR_BGR2GRAY:
                if hasattr(src, 'shape') and len(src.shape) == 3:
                    return np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)
            return np.zeros_like(src)
            
        def inRange(self, src, lowerb, upperb):
            """Stub for cv2.inRange"""
            if hasattr(src, 'shape'):
                if len(src.shape) == 3:
                    return np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)
                return np.zeros(src.shape[:2], dtype=np.uint8)
            return np.zeros((1, 1), dtype=np.uint8)
            
        def morphologyEx(self, src, op, kernel, dst=None, anchor=None, 
                        iterations=None, borderType=None, borderValue=None):
            """Stub for cv2.morphologyEx"""
            return np.zeros_like(src)
            
        def dilate(self, src, kernel, dst=None, anchor=None, iterations=1,
                borderType=None, borderValue=None):
            """Stub for cv2.dilate"""
            return np.zeros_like(src)
            
        def erode(self, src, kernel, dst=None, anchor=None, iterations=1,
                borderType=None, borderValue=None):
            """Stub for cv2.erode"""
            return np.zeros_like(src)
            
        def findContours(self, image, mode, method, contours=None, 
                        hierarchy=None, offset=None):
            """Stub for cv2.findContours"""
            return [], None
            
        def contourArea(self, contour, oriented=None):
            """Stub for cv2.contourArea"""
            return 0.0
            
        def minAreaRect(self, points):
            """Stub for cv2.minAreaRect"""
            return ((0, 0), (0, 0), 0)
            
        def boxPoints(self, box):
            """Stub for cv2.boxPoints"""
            return np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
            
        def convexHull(self, points, clockwise=None, returnPoints=True):
            """Stub for cv2.convexHull"""
            return points  # Just return the input points
            
        def drawContours(self, image, contours, contourIdx, color, 
                        thickness=None, lineType=None, hierarchy=None, 
                        maxLevel=None, offset=None):
            """Stub for cv2.drawContours"""
            return None
            
        def bitwise_and(self, src1, src2, dst=None, mask=None):
            """Stub for cv2.bitwise_and"""
            if hasattr(src1, 'shape'):
                return np.zeros_like(src1)
            return np.zeros((1, 1), dtype=np.uint8)
            
        def bitwise_or(self, src1, src2, dst=None, mask=None):
            """Stub for cv2.bitwise_or"""
            if hasattr(src1, 'shape'):
                return np.zeros_like(src1)
            return np.zeros((1, 1), dtype=np.uint8)
            
        def threshold(self, src, thresh, maxval, type, dst=None):
            """Stub for cv2.threshold"""
            if hasattr(src, 'shape'):
                return 0, np.zeros_like(src)
            return 0, np.zeros((1, 1), dtype=np.uint8)
            
        def GaussianBlur(self, src, ksize, sigmaX, sigmaY=0, borderType=None):
            """Stub for cv2.GaussianBlur"""
            return src
            
        def Canny(self, image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None):
            """Stub for cv2.Canny"""
            if hasattr(image, 'shape'):
                if len(image.shape) == 3:
                    return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                return np.zeros(image.shape, dtype=np.uint8)
            return np.zeros((1, 1), dtype=np.uint8)
            
        def HoughLinesP(self, image, rho, theta, threshold, lines=None, 
                       minLineLength=None, maxLineGap=None):
            """Stub for cv2.HoughLinesP"""
            return np.array([[[0, 0, 0, 0]]], dtype=np.int32)
            
        def line(self, img, pt1, pt2, color, thickness=1, lineType=8, shift=0):
            """Stub for cv2.line"""
            return img
    
    # Replace cv2 with our comprehensive stub
    cv2 = CV2Stub()

class FieldDetector:
    """Detects the soccer field boundaries and filters players accordingly"""
    
    def __init__(self):
        self.field_mask = None
        self.field_contour = None
        self.detection_history = []  # Store previous detections for stability
        self.field_detected = False
        self.detection_confidence = 0
        
        # Field detection parameters
        self.min_field_area_ratio = 0.15  # Field must be at least 15% of frame
        self.history_size = 5  # Number of previous detections to keep
    
    def detect_field(self, frame):
        """
        Detect the soccer field boundaries in a frame
        
        Args:
            frame: Input image frame
            
        Returns:
            (field_mask, boundary_points): Binary mask of field and boundary points
        """
        try:
            # If we already have a confident field detection, only update occasionally
            if self.field_detected and self.detection_confidence > 5:
                if len(self.detection_history) % 30 != 0:
                    return self.field_mask, self.field_contour
            
            # ------------------------------------
            # 1. GREEN FIELD DETECTION (PRIMARY)
            # ------------------------------------
            
            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Try multiple green ranges to handle different lighting/fields
            green_masks = []
            
            # Standard grass green
            lower_green1 = np.array([35, 40, 40])
            upper_green1 = np.array([85, 255, 255])
            green_mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
            green_masks.append(green_mask1)
            
            # Darker grass green
            lower_green2 = np.array([30, 30, 30])
            upper_green2 = np.array([90, 255, 255])
            green_mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
            green_masks.append(green_mask2)
            
            # Lighter/yellower grass
            lower_green3 = np.array([25, 25, 40])
            upper_green3 = np.array([95, 255, 255])
            green_mask3 = cv2.inRange(hsv, lower_green3, upper_green3)
            green_masks.append(green_mask3)
            
            # Combine all green masks
            green_mask = cv2.bitwise_or(green_mask1, green_mask2)
            green_mask = cv2.bitwise_or(green_mask, green_mask3)
            
            # Clean up the mask with morphological operations
            kernel_close = np.ones((15, 15), np.uint8)  # Larger kernel to close gaps
            kernel_open = np.ones((5, 5), np.uint8)     # Smaller kernel to remove noise
            
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel_close)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel_open)
            
            # ------------------------------------
            # 2. WHITE LINE DETECTION (SECONDARY)
            # ------------------------------------
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 1)
            
            # Adaptive thresholding to find white lines
            _, white_mask = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)
            
            # Dilate to connect broken lines
            kernel_dilate = np.ones((3, 3), np.uint8)
            white_mask = cv2.dilate(white_mask, kernel_dilate, iterations=1)
            
            # Create a combined field and line mask
            # White lines on field should be part of field
            field_and_lines = cv2.bitwise_or(green_mask, white_mask)
            
            # ------------------------------------
            # 3. CONTOUR ANALYSIS
            # ------------------------------------
            
            # Find contours from the green field mask
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # If no contours found, use previous detection if available
                if self.field_mask is not None:
                    return self.field_mask, self.field_contour
                return None, None
            
            # Calculate minimum area threshold (% of frame)
            min_area = frame.shape[0] * frame.shape[1] * self.min_field_area_ratio
            
            # Filter by area and get the largest contour
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            if not large_contours:
                if self.field_mask is not None:
                    return self.field_mask, self.field_contour
                return None, None
            
            # Get the largest contour (which should be the field)
            field_contour = max(large_contours, key=cv2.contourArea)
            
            # ------------------------------------
            # 4. CONTOUR REFINEMENT
            # ------------------------------------
            
            # Create a smoother boundary with convex hull
            try:
                hull = cv2.convexHull(field_contour)
            except Exception:
                hull = field_contour  # Fall back to original contour if hull fails
            
            # Create a field mask from the hull
            field_mask = np.zeros_like(green_mask)
            cv2.drawContours(field_mask, [hull], 0, 255, -1)
            
            # Create a binary mask
            _, field_mask = cv2.threshold(field_mask, 127, 255, cv2.THRESH_BINARY)
            
            # ------------------------------------
            # 5. TEMPORAL CONSISTENCY
            # ------------------------------------
            
            # Store the detection
            self.field_mask = field_mask
            self.field_contour = hull
            self.detection_history.append(hull)
            
            # Keep history manageable
            if len(self.detection_history) > self.history_size:
                self.detection_history.pop(0)
            
            # Increase detection confidence
            self.detection_confidence += 1
            self.field_detected = True
            
            return field_mask, hull
            
        except Exception as e:
            print(f"Error in field detection: {e}")
            # Fall back to previous detection if available
            if self.field_mask is not None:
                return self.field_mask, self.field_contour
            return None, None
    
    def filter_players(self, bounding_boxes):
        """Filter out detections outside the field"""
        if self.field_mask is None:
            return bounding_boxes
            
        filtered_boxes = []
        for box in bounding_boxes:
            x, y, w, h = box
            # Use the feet (bottom center) as the reference point
            feet_x, feet_y = int(x + w//2), int(y + h)
            
            # Additional points to check (to avoid filtering out valid players at the edge)
            points_to_check = [
                (feet_x, feet_y),  # Feet/bottom center
                (x + w//2, y + h - h//4),  # Lower body
                (x + w//4, y + h - h//4),  # Lower left
                (x + 3*w//4, y + h - h//4)  # Lower right
            ]
            
            on_field = False
            for px, py in points_to_check:
                # Ensure coordinates are within bounds
                if (0 <= py < self.field_mask.shape[0] and 
                    0 <= px < self.field_mask.shape[1]):
                    # Check if point is on field (mask value > 0)
                    if self.field_mask[py, px] > 0:
                        on_field = True
                        break
            
            if on_field:
                filtered_boxes.append(box)
                
        return filtered_boxes