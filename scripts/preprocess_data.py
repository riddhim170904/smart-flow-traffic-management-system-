import cv2
import numpy as np
import os
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import json
from datetime import datetime

class DataPreprocessor:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        target_size: Tuple[int, int] = (640, 640),
        augment: bool = True
    ):
        """
        Initialize the data preprocessor.
        
        Args:
            input_dir: Directory containing raw images/videos
            output_dir: Directory to save processed data
            target_size: Target size for resizing images (width, height)
            augment: Whether to apply data augmentation
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.augment = augment
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def process_image(
        self,
        image_path: Path,
        annotation_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Process a single image and its annotation.
        
        Args:
            image_path: Path to the input image
            annotation_path: Path to the annotation file (optional)
            
        Returns:
            Tuple of (processed_image, annotation_data)
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
                
            # Resize image
            image = cv2.resize(image, self.target_size)
            
            # Read annotation if provided
            annotation_data = None
            if annotation_path and annotation_path.exists():
                with open(annotation_path, 'r') as f:
                    annotation_data = json.load(f)
                    
            return image, annotation_data
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            raise
            
    def process_video(
        self,
        video_path: Path,
        output_fps: int = 10,
        max_frames: Optional[int] = None
    ) -> List[Path]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the input video
            output_fps: Frames per second to extract
            max_frames: Maximum number of frames to extract (optional)
            
        Returns:
            List of paths to extracted frames
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / output_fps)
            
            frame_paths = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    # Process and save frame
                    frame = cv2.resize(frame, self.target_size)
                    frame_path = self.output_dir / "images" / f"{video_path.stem}_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(frame_path)
                    
                    if max_frames and len(frame_paths) >= max_frames:
                        break
                        
                frame_count += 1
                
            cap.release()
            return frame_paths
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")
            raise
            
    def process_directory(self) -> None:
        """Process all images and videos in the input directory."""
        try:
            # Process images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            for image_path in self.input_dir.glob('**/*'):
                if image_path.suffix.lower() in image_extensions:
                    self.logger.info(f"Processing image: {image_path}")
                    processed_image, annotation = self.process_image(image_path)
                    
                    # Save processed image
                    output_path = self.output_dir / "images" / image_path.name
                    cv2.imwrite(str(output_path), processed_image)
                    
                    # Save annotation if available
                    if annotation:
                        annotation_path = self.output_dir / "annotations" / f"{image_path.stem}.json"
                        with open(annotation_path, 'w') as f:
                            json.dump(annotation, f, indent=2)
                            
            # Process videos
            video_extensions = {'.mp4', '.avi', '.mov'}
            for video_path in self.input_dir.glob('**/*'):
                if video_path.suffix.lower() in video_extensions:
                    self.logger.info(f"Processing video: {video_path}")
                    self.process_video(video_path)
                    
        except Exception as e:
            self.logger.error(f"Error processing directory: {e}")
            raise
            
    def generate_dataset_info(self) -> None:
        """Generate dataset information and statistics."""
        try:
            dataset_info = {
                'name': 'SmartFlow Traffic Dataset',
                'created': datetime.now().isoformat(),
                'image_count': len(list((self.output_dir / "images").glob('*.jpg'))),
                'annotation_count': len(list((self.output_dir / "annotations").glob('*.json'))),
                'target_size': self.target_size,
                'augmented': self.augment
            }
            
            # Save dataset info
            info_path = self.output_dir / "dataset_info.json"
            with open(info_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)
                
            self.logger.info(f"Dataset info saved to: {info_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating dataset info: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(
        input_dir="data/raw",
        output_dir="data/processed",
        target_size=(640, 640),
        augment=True
    )
    preprocessor.process_directory()
    preprocessor.generate_dataset_info() 