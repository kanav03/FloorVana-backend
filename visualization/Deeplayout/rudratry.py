import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum
import os

# Constants matching C++ implementation
HALFDOORWIDTH = 6
HALFMIDWINDOWWIDTH = 9
HALFSMALLWINDOWWIDTH = 5
HALFLARGEWINDOWWIDTH = 12

class RoomType(Enum):
    PRIVATE = 0
    PUBLIC = 1

@dataclass
class RoomLabel:
    index: int
    name: str
    type: RoomType
    texture_path: str

# Room label definitions matching C++ ROOMLABEL array
ROOM_LABELS = [
    RoomLabel(0, "LivingRoom", RoomType.PUBLIC, "PublicArea"),
    RoomLabel(1, "MasterRoom", RoomType.PRIVATE, "Bedroom"),
    RoomLabel(2, "Kitchen", RoomType.PUBLIC, "FunctionArea"),
    RoomLabel(3, "Bathroom", RoomType.PRIVATE, "FunctionArea"),
    RoomLabel(4, "DiningRoom", RoomType.PUBLIC, "FunctionArea"),
    RoomLabel(5, "ChildRoom", RoomType.PRIVATE, "Bedroom"),
    RoomLabel(6, "StudyRoom", RoomType.PRIVATE, "Bedroom"),
    RoomLabel(7, "SecondRoom", RoomType.PRIVATE, "Bedroom"),
    RoomLabel(8, "GuestRoom", RoomType.PRIVATE, "Bedroom"),
    RoomLabel(9, "Balcony", RoomType.PUBLIC, "PublicArea"),
    RoomLabel(10, "Entrance", RoomType.PUBLIC, "PublicArea"),
    RoomLabel(11, "Storage", RoomType.PRIVATE, "PublicArea"),
    RoomLabel(12, "Wall-in", RoomType.PRIVATE, "PublicArea")
]

class TextureManager:
    def __init__(self):
        self.textures: Dict[str, np.ndarray] = {}
        self._load_textures()
    
    def _load_textures(self):
        texture_dir = "texture"
        for label in ROOM_LABELS:
            texture_path = os.path.join(texture_dir, f"{label.texture_path}.jpg")
            if os.path.exists(texture_path):
                texture = cv2.imread(texture_path)
                if texture is not None:
                    # Resize texture to standard size
                    texture = cv2.resize(texture, (256, 256))
                    self.textures[label.name] = texture
                    
    def get_texture(self, room_name: str) -> np.ndarray:
        return self.textures.get(room_name, np.ones((256,256,3), dtype=np.uint8)*245)

def apply_room_texture(img: np.ndarray, points: np.ndarray, texture: np.ndarray, alpha: float = 0.7):
    """Apply texture to room while preserving boundaries"""
    # Create mask for the room
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    
    # Get room bounds
    x, y, w, h = cv2.boundingRect(points)
    
    # Resize texture to fit room dimensions
    room_texture = cv2.resize(texture, (w, h))
    
    # Create textured room image
    textured_room = np.zeros_like(img)
    textured_room[y:y+h, x:x+w] = room_texture
    
    # Create 3-channel mask
    mask_3ch = np.stack([mask, mask, mask], axis=-1) / 255.0
    
    # Apply blending only where mask is non-zero
    blend = np.where(
        mask_3ch > 0,
        cv2.addWeighted(
            img.astype(float),
            1-alpha,
            textured_room.astype(float),
            alpha,
            0
        ),
        img
    )
    
    # Update the image with the blended result
    img[:] = blend.astype(np.uint8)

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
    """Enhanced save_floor_plan with textures and improved styling"""
    # Initialize texture manager
    texture_mgr = TextureManager()
    
    # Create high resolution image
    img = np.ones((4096, 4096, 3), dtype=np.uint8) * 255
    
    # Scale factor and margins
    scale = 8
    margin = 100
    
    # Calculate bounds
    points = np.array(house.exterior_wall)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # Calculate scaling to fit in image with margins
    scale_x = (img.shape[1] - 2*margin) / width
    scale_y = (img.shape[0] - 2*margin) / height
    scale = min(scale_x, scale_y)
    
    # Transform function to scale and center points
    def transform_points(pts):
        pts = np.array(pts) * scale
        pts += [margin, margin]
        return pts.astype(np.int32)
    
    # Draw rooms with textures
    for room in house.rooms:
        points = transform_points(room.boundary)
        texture = texture_mgr.get_texture(room.name)
        apply_room_texture(img, points, texture)
        
        # Draw room boundary
        cv2.polylines(img, [points], True, (100,100,100), 3)
        
        # Add room label
        centroid = points.mean(axis=0).astype(int)
        font_scale = scale/100
        cv2.putText(img, room.name, tuple(centroid), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale,
                   (0,0,0), max(1, int(scale/30)))
    
    # Draw exterior walls
    ext_wall_pts = transform_points(house.exterior_wall)
    cv2.polylines(img, [ext_wall_pts], True, (0,0,0), max(3, int(scale/20)))
    
    # Draw interior walls
    for wall in house.interior_walls:
        wall_pts = transform_points(wall)
        cv2.polylines(img, [wall_pts], True, (80,80,80), max(2, int(scale/30)))
    
    # Draw doors with improved styling
    def draw_door(x, y, w, h, is_main=False):
        pts = transform_points([(x,y), (x+w,y), (x+w,y+h), (x,y+h)])
        # Door frame
        cv2.polylines(img, [pts], True, (0,0,0), max(2, int(scale/40)))
        # Door arc
        center = tuple(transform_points([(x+w/2, y)])[0])
        axes = (int(w*scale/2), int(h*scale/2))
        cv2.ellipse(img, center, axes, 0, 0, 180, (0,0,0), max(2, int(scale/40)))
        if is_main:
            # Add handle
            handle_pos = tuple(transform_points([(x+w*0.8, y+h/2)])[0])
            cv2.circle(img, handle_pos, max(3, int(scale/30)), (0,0,0), -1)
    
    # Draw main door
    x, y, w, h = house.front_door
    draw_door(x, y, w, h, is_main=True)
    
    # Draw room doors and windows
    for room in house.rooms:
        if room.entry:
            x, y, w, h = room.entry            
            draw_door(x, y, w, h)
        
        # Draw windows
        for window in room.windows:
            x, y, w, h = window
            pts = transform_points([(x,y), (x+w,y), (x+w,y+h), (x,y+h)])
            cv2.polylines(img, [pts], True, (150,150,150), max(2, int(scale/40)))
            # Window panes
            mid_x = (pts[0][0] + pts[2][0])//2
            mid_y = (pts[0][1] + pts[2][1])//2
            cv2.line(img, (mid_x, pts[0][1]), (mid_x, pts[2][1]), (150,150,150), 1)
            cv2.line(img, (pts[0][0], mid_y), (pts[2][0], mid_y), (150,150,150), 1)
    
    # Add final touches
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