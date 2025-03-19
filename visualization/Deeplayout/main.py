import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Room:
    name: str
    boundary: List[Tuple[int, int]]
    entry: Tuple[int, int, int, int]  # x, y, width, height
    windows: List[Tuple[int, int, int, int]]

@dataclass
class House:
    exterior_wall: List[Tuple[int, int]]
    interior_walls: List[List[Tuple[int, int]]]
    front_door: Tuple[int, int, int, int]
    rooms: List[Room]

def process_floor_plan(image_path: str) -> House:
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    
    # Add image size check
    if img.size == 0:
        raise ValueError("Image is empty")
        
    print(f"Image loaded successfully. Shape: {img.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Add Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Improve wall detection with adaptive thresholding
    walls = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Optional: Add morphological operations to clean up the image
    kernel = np.ones((3,3), np.uint8)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel)
    
    # Find contours with different mode
    contours, _ = cv2.findContours(walls, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} contours")
    
    # Process exterior walls
    exterior_wall = []
    interior_walls = []  # Initialize interior_walls list
    
    if not contours:
        raise ValueError("No contours found in the image")
        
    ext_contour = max(contours, key=cv2.contourArea)
    ext_contour = ext_contour.reshape(-1, 2)
    exterior_wall = [(int(point[0]), int(point[1])) for point in ext_contour]
    
    # Process interior walls
    for contour in contours:
        if not np.array_equal(contour, ext_contour.reshape(-1, 1, 2)):
            wall = []
            contour = contour.reshape(-1, 2)
            for point in contour:
                wall.append((int(point[0]), int(point[1])))
            if wall:
                interior_walls.append(wall)
    
    # Detect rooms
    rooms = []
    room_mask = np.zeros_like(gray)
    cv2.drawContours(room_mask, contours, -1, 255, -1)
    
    # Improved room detection
    labeled = cv2.connectedComponents(room_mask)[1]
    unique_labels = np.unique(labeled)[1:]  # Skip background label 0
    
    # Constants for door and window sizes (from deeplayout.cpp)
    HALF_DOOR_WIDTH = 20
    HALF_SMALL_WINDOW_WIDTH = 20 
    HALF_MID_WINDOW_WIDTH = 30
    HALF_LARGE_WINDOW_WIDTH = 40

    # Room detection with improved boundary handling
    rooms = []
    room_mask = np.zeros_like(gray)
    cv2.drawContours(room_mask, contours, -1, 255, -1)
    
    labeled = cv2.connectedComponents(room_mask)[1]
    unique_labels = np.unique(labeled)[1:]  # Skip background label 0
    
    # Use distance transform for better room separation
    dist = cv2.distanceTransform(room_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Improved room labeling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Room type detection based on size and aspect ratio
    room_types = {
        0: "LivingRoom",
        1: "MasterRoom",
        2: "Bathroom",
        3: "StudyRoom"
    }
    
    for room_id in range(1, markers.max() + 1):
        room_binary = np.uint8(labeled == room_id)
        room_contours, _ = cv2.findContours(room_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if room_contours:
            room_contour = max(room_contours, key=cv2.contourArea)
            if cv2.contourArea(room_contour) > 100:  # Minimum room size
                # Get room properties
                x, y, w, h = cv2.boundingRect(room_contour)
                centroid = (x + w//2, y + h//2)
                
                # Improved door placement based on room type
                if f"Room_{room_id}" in ["Bathroom", "MasterRoom", "StudyRoom"]:
                    # Place door on right wall
                    door = (x + w, y + h//2 - HALF_DOOR_WIDTH, 40, HALF_DOOR_WIDTH * 2)
                else:
                    # Place door on bottom wall
                    door = (x + w//2 - HALF_DOOR_WIDTH, y + h, HALF_DOOR_WIDTH * 2, 40)

                # Smart window placement based on exterior walls
                windows = []
                if w > h:  # Wide room - windows on longer walls
                    windows = [
                        (x + w//4, y, HALF_MID_WINDOW_WIDTH * 2, 40),  # Left window
                        (x + 3*w//4, y, HALF_MID_WINDOW_WIDTH * 2, 40)  # Right window
                    ]
                else:  # Tall room - windows on shorter walls
                    windows = [
                        (x, y + h//4, 40, HALF_MID_WINDOW_WIDTH * 2),  # Top window
                        (x, y + 3*h//4, 40, HALF_MID_WINDOW_WIDTH * 2)  # Bottom window
                    ]

                # Create room with improved boundary detection
                boundary = [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]
                room = Room(
                    name=room_types.get(room_id-1, f"Room_{room_id}"),
                    boundary=boundary,
                    entry=door,
                    windows=windows
                )
                rooms.append(room)

    # Detect front door with better placement
    if exterior_wall:
        # Find the bottom-most point of exterior wall
        bottom_point = max(exterior_wall, key=lambda p: p[1])
        front_door = (bottom_point[0] - HALF_DOOR_WIDTH, bottom_point[1] - 40, 
                     HALF_DOOR_WIDTH * 2, 40)
    else:
        front_door = (0, img.shape[0]//2, 40, 40)  # Default fallback

    # Add debug printing
    print("\n=== Floor Plan Debug Information ===")
    
    print("\nExterior Wall Points:")
    for i, point in enumerate(exterior_wall):
        print(f"Point {i}: ({point[0]}, {point[1]})")
    
    print("\nInterior Walls:")
    for i, wall in enumerate(interior_walls):
        print(f"\nWall {i} points:")
        for j, point in enumerate(wall):
            print(f"  Point {j}: ({point[0]}, {point[1]})")

    print("\nRooms:")
    for room in rooms:
        print(f"\nRoom: {room.name}")
        print("  Boundary points:")
        for i, point in enumerate(room.boundary):
            print(f"    Point {i}: ({point[0]}, {point[1]})")
        print("  Entry door: (x={}, y={}, w={}, h={})".format(*room.entry))
        print("  Windows:")
        for i, window in enumerate(room.windows):
            print(f"    Window {i}: (x={window[0]}, y={window[1]}, w={window[2]}, h={window[3]})")

    print("\nFront Door: (x={}, y={}, w={}, h={})".format(*front_door))
    print("\n================================")

    # Create house with improved properties
    house = House(
        exterior_wall=exterior_wall,
        interior_walls=interior_walls,
        front_door=front_door,
        rooms=rooms
    )
    
    return house

def add_interior_door(house: House) -> House:
    """Add interior doors using smart placement rules"""
    # Add interior doors between rooms
    for room in house.rooms:
        if room.name != "LivingRoom":
            # Find closest wall to living room
            # Add door based on room type
            pass
    return house

def add_windows(house: House) -> House:
    """Add windows using intelligent placement"""
    for room in house.rooms:
        # Find exterior walls
        # Add appropriate sized windows
        pass
    return house

def save_floor_plan(house: House, output_path: str):
    # Create higher resolution image with white background
    img = np.zeros((2048, 2048, 3), dtype=np.uint8)
    img.fill(255)
    
    # Scale factor for better visualization
    scale = 2
    
    # Draw rooms first (as filled polygons)
    for room in house.rooms:
        # Scale points
        pts = np.array([(p[0]*scale, p[1]*scale) for p in room.boundary], np.int32)
        pts = pts.reshape((-1,1,2))
        
        # Fill room with light color based on room type
        room_colors = {
            "Bathroom": (255, 240, 240),  # Light pink
            "MasterRoom": (240, 255, 240),  # Light green
            "StudyRoom": (240, 240, 255),  # Light blue
            "LivingRoom": (255, 255, 240),  # Light yellow
        }
        color = room_colors.get(room.name, (245, 245, 245))  # Default light gray
        cv2.fillPoly(img, [pts], color)
        
        # Draw room boundary with thicker lines
        cv2.polylines(img, [pts], True, (100,100,100), 3)
        
        # Add room label
        centroid = np.mean(pts, axis=0, dtype=np.int32)
        cv2.putText(img, room.name, tuple(centroid[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        
        # Draw entry doors with better styling
        x, y, w, h = [val*scale for val in room.entry]
        # Draw door arc
        center = (x + w//2, y)
        axes = (w//2, h//2)
        cv2.ellipse(img, center, axes, 0, 0, 180, (0,100,0), 2)
        
        # Draw windows with 3D effect
        for window in room.windows:
            x, y, w, h = [val*scale for val in window]
            # Main window frame
            cv2.rectangle(img, (x,y), (x+w,y+h), (150,150,150), 2)
            # Window panes
            cv2.line(img, (x+w//2,y), (x+w//2,y+h), (150,150,150), 1)
            cv2.line(img, (x,y+h//2), (x+w,y+h//2), (150,150,150), 1)
            # 3D effect
            cv2.line(img, (x,y), (x-3,y-3), (100,100,100), 1)
            cv2.line(img, (x+w,y), (x+w-3,y-3), (100,100,100), 1)
    
    # Draw exterior walls with thick lines
    if house.exterior_wall:
        pts = np.array([(p[0]*scale, p[1]*scale) for p in house.exterior_wall], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,0,0), 4)
    
    # Draw interior walls with better styling
    for wall in house.interior_walls:
        if wall:
            pts = np.array([(p[0]*scale, p[1]*scale) for p in wall], np.int32)
            pts = pts.reshape((-1,1,2))
            # Draw main wall line
            cv2.polylines(img, [pts], True, (80,80,80), 3)
            # Add shadow effect
            cv2.polylines(img, [pts + 2], True, (150,150,150), 1)
    
    # Draw front door with special styling
    x, y, w, h = [val*scale for val in house.front_door]
    # Door frame
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,0), 3)
    # Door handle
    handle_x = x + int(w*0.8)
    handle_y = y + h//2
    cv2.circle(img, (handle_x, handle_y), 3, (0,0,0), -1)
    # Door label
    cv2.putText(img, "MAIN", (x-20, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    
    # Add some post-processing effects
    img = cv2.GaussianBlur(img, (3,3), 0)
    
    # Save high-quality image
    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def main():
    # Set input and output paths
    input_path = "synth_normalization/0in1433.png"  # Replace with your image path
    output_path = "meow.png"
    
    try:
        # Process the image and generate floor plan
        house = process_floor_plan(input_path)
        
        # Save the floor plan
        save_floor_plan(house, output_path)
        print(f"Floor plan generated and saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing floor plan: {str(e)}")

if __name__ == "__main__":
    main()