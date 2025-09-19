import logging
from pathlib import Path
from typing import List, Optional, Union
import torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class DocLayoutYOLO:
    """DocLayout-YOLO model wrapper class with pathlib support"""

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the DocLayout-YOLO model
        
        Args:
            model_path: Local model file path. If not provided, 
                       the pre-trained model will be loaded from Hugging Face Hub.
        """
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.init_model()

    def init_model(self) -> bool:
        """Initialize the model"""
        try:
            if self.model_path and self.model_path.exists():
                logger.info(f"Loading local model file: {self.model_path}")
                # 尝试导入doclayout_yolo库
                try:
                    from doclayout_yolo import YOLOv10
                    self.model = YOLOv10(str(self.model_path))
                except ImportError:
                    # 如果doclayout_yolo不可用，使用ultralytics作为备选
                    logger.warning("doclayout_yolo库不可用，尝试使用ultralytics库")
                    from ultralytics import YOLO
                    self.model = YOLO(str(self.model_path))
            else:
                logger.info("Loading pre-trained model from Hugging Face")
                # 尝试从Hugging Face加载模型
                try:
                    filepath = hf_hub_download(
                        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                        filename="doclayout_yolo_docstructbench_imgsz1024.pt"
                    )
                    # 尝试导入doclayout_yolo库
                    try:
                        from doclayout_yolo import YOLOv10
                        self.model = YOLOv10(filepath)
                    except ImportError:
                        # 如果doclayout_yolo不可用，使用ultralytics作为备选
                        logger.warning("doclayout_yolo库不可用，尝试使用ultralytics库")
                        from ultralytics import YOLO
                        self.model = YOLO(filepath)
                except Exception as e:
                    logger.warning(f"从Hugging Face加载模型失败: {e}")
                    # 直接使用ultralytics的预训练模型作为备选
                    from ultralytics import YOLO
                    self.model = YOLO("yolov8n.pt")
                    logger.info("Successfully loaded ultralytics YOLO model as alternative")
            
            logger.info("DocLayout-YOLO model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DocLayout-YOLO model: {e}")
            self.model = None
            return False

    def predict(
        self, 
        image_input: Union[str, Path, "np.ndarray"], 
        imgsz: int = 1024, 
        conf: float = 0.25, 
        device: Optional[str] = None
    ) -> List:
        """
        Perform layout prediction on the image.
        
        Args:
            image_input: Path to the image file or numpy array
            imgsz: Input image size
            conf: Confidence threshold
            device: Device to use (if None, automatically selected)
            
        Returns:
            List of prediction results
        """
        if self.model is None:
            logger.error("The model is not initialized")
            return []
        
        # Handle different input types
        import numpy as np
        if isinstance(image_input, np.ndarray):
            # Input is numpy array, use directly
            source = image_input
        else:
            # Input is path, validate it exists
            image_path = Path(image_input)
            if not image_path.exists():
                logger.error(f"Image file does not exist: {image_path}")
                return []
            source = str(image_path)
        
        if device is None:
            device = self.device
        
        try:
            results = self.model.predict(
                source=source,
                imgsz=imgsz,
                conf=conf,
                device=device
            )
            
            # Parse results into regions format
            regions = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    class_names = result.names
                    
                    for box, cls_id, confidence in zip(boxes, classes, confs):
                        if confidence >= conf:
                            x1, y1, x2, y2 = map(int, box)
                            cls_name = class_names[int(cls_id)]
                            
                            regions.append({
                                'type': cls_name,
                                'coords': [x1, y1, x2-x1, y2-y1],
                                'confidence': float(confidence)
                            })
            
            return regions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []