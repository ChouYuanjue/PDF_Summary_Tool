"""
Base LLM API client with support for multiple backends including Gemini and SiliconCloud.
Provides a unified interface for text extraction, special content analysis, and text correction.
"""

import abc
import difflib
import gc
import io
import json
import logging
import os
import time
import re
import threading
from typing import Any, Dict, Optional

# 尝试从.env文件加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("未安装python-dotenv，无法从.env文件加载环境变量")

import cv2
import numpy as np
import requests
from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class BaseLLMClient(abc.ABC):
    """Base class for LLM API clients"""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize LLM client
        
        Args:
            model: Model name to use
            api_key: API key (if not provided, reads from environment)
        """
        self.model = model
        self.api_key = api_key
        self.client = None
        
    @abc.abstractmethod
    def _setup_client(self):
        """Setup API client"""
        pass
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if API client is available"""
        pass
    
    @abc.abstractmethod
    def extract_text(self, region_img: np.ndarray, region_info: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Extract text from region using LLM API"""
        pass
    
    @abc.abstractmethod
    def process_special_region(self, region_img: np.ndarray, region_info: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Process special regions (tables, figures) with LLM API"""
        pass
    
    @abc.abstractmethod
    def correct_text(self, text: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Correct OCR text using LLM API"""
        pass
    
    def reload_client(self, api_key: Optional[str] = None) -> bool:
        """Reload the API client"""
        if api_key:
            self.api_key = api_key
        self.client = self._setup_client()
        return self.is_available()
    
    def _resize_image_if_needed(self, image: np.ndarray, max_dim: int = 1024) -> np.ndarray:
        """Resize image if it exceeds max dimension"""
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h))
        return image
    
    def _convert_image_to_bytes(self, image: np.ndarray, format: str = 'JPEG', quality: int = 85) -> bytes:
        """Convert numpy array image to bytes"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format=format, quality=quality, optimize=True)
        img_bytes = img_byte_arr.getvalue()
        
        del pil_image, img_byte_arr
        gc.collect()
        
        return img_bytes


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini VLM API client implementation"""
    
    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """Initialize Gemini API client"""
        super().__init__(model, api_key)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = self._setup_client()
    
    def _setup_client(self) -> Optional[genai.Client]:
        """Setup Gemini API client"""
        try:
            if not self.api_key:
                logger.warning("GEMINI_API_KEY environment variable not set")
                return None
            
            client = genai.Client(api_key=self.api_key)
            logger.info("Gemini API client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API client: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Gemini API client is available"""
        return self.client is not None
    
    def extract_text(self, region_img: np.ndarray, region_info: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Extract text from region using Gemini API"""
        if not self.is_available():
            logger.warning("Gemini API client not initialized")
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'text': '',
                'confidence': 0.0
            }
        
        try:
            # Resize image if too large
            region_img_resized = self._resize_image_if_needed(region_img)
            img_bytes = self._convert_image_to_bytes(region_img_resized)

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                    ],
                ),
            ]

            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
            )

            logger.info(f"Requesting Gemini extract_text (model={self.model})")
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            )
            
            del region_img_resized, img_bytes
            gc.collect()
            
            text = response.text.strip()
            
            result = {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'text': text,
                'confidence': region_info.get('confidence', 1.0)
            }
            
            del response
            gc.collect()
            
            return result
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Gemini text extraction error: {e}")
            
            # Handle rate limit errors specifically
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                logger.error(f"Rate limit exceeded: {error_str}")
                return {
                    'type': region_info['type'],
                    'coords': region_info['coords'],
                    'text': '[RATE_LIMIT_EXCEEDED]',
                    'confidence': 0.0,
                    'error': 'gemini_rate_limit',
                    'error_message': error_str  # 返回完整的错误信息，包含建议的等待时间
                }
            
            # Handle other Gemini API errors
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'text': '[GEMINI_EXTRACTION_FAILED]',
                'confidence': 0.0,
                'error': 'gemini_api_error',
                'error_message': str(e)
            }
    
    def process_special_region(self, region_img: np.ndarray, region_info: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Process special regions (tables, figures) with Gemini API"""
        if not self.is_available():
            logger.warning("Gemini API client not initialized")
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'content': 'Gemini API not available',
                'analysis': 'Client not initialized',
                'confidence': 0.0
            }
        
        try:
            # Resize image if too large
            region_img_resized = self._resize_image_if_needed(region_img)
            img_bytes = self._convert_image_to_bytes(region_img_resized)

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                    ],
                ),
            ]

            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
            )

            logger.info(f"Requesting Gemini process_special_region (model={self.model})")
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            )
            
            del region_img_resized, img_bytes
            gc.collect()
            
            response_text = response.text.strip()
            parsed_result = self._parse_gemini_response(response_text, region_info)
            
            del response
            gc.collect()
            
            return parsed_result
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Gemini special region processing error: {e}")
            
            # Handle rate limit errors
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                return {
                    'type': region_info['type'],
                    'coords': region_info['coords'],
                    'content': '[RATE_LIMIT_EXCEEDED]',
                    'analysis': 'Rate limit exceeded',
                    'confidence': 0.0,
                    'error': 'gemini_rate_limit',
                    'error_message': 'Gemini API rate limit exceeded'
                }
            
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'content': '[GEMINI_PROCESSING_FAILED]',
                'analysis': f'Processing failed: {str(e)}',
                'confidence': 0.0,
                'error': 'gemini_api_error',
                'error_message': str(e)
            }
    
    def correct_text(self, text: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Correct OCR text using Gemini API"""
        if not self.is_available() or not text:
            return {"corrected_text": text, "confidence": 0.0}
        
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"{system_prompt}\n\n{user_prompt}")],
                ),
            ]

            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
            )

            logger.info(f"Requesting Gemini correct_text (model={self.model})")
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            )

            corrected_text = response.text.strip()
            
            sm = difflib.SequenceMatcher(None, text, corrected_text)
            confidence = sm.ratio()

            return {
                "corrected_text": corrected_text,
                "confidence": confidence
            }

        except Exception as e:
            error_str = str(e)
            logger.error(f"Text correction error: {e}")
            
            # Handle rate limit errors specifically
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                logger.error("Rate limit exceeded during text correction")
                return {"corrected_text": text, "confidence": 0.0, "error": "rate_limit"}
            
            # Handle service unavailable errors
            elif "503" in error_str or "UNAVAILABLE" in error_str:
                logger.error("Service unavailable during text correction")
                return {"corrected_text": text, "confidence": 0.0, "error": "service_unavailable"}
            
            # For other errors, return original text with error indicator
            else:
                logger.error("Text correction failed with other error")
                return {"corrected_text": text, "confidence": 0.0, "error": "correction_failed"}
    
    def _parse_gemini_response(self, response_text: str, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Gemini response for special regions"""
        try:
            parsed = json.loads(response_text)
            
            result = {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'confidence': region_info.get('confidence', 1.0)
            }
            
            if region_info['type'] == 'table':
                result['content'] = parsed.get('markdown_table', '')
                result['analysis'] = parsed.get('summary', '')
                result['educational_value'] = parsed.get('educational_value', '')
                result['related_topics'] = parsed.get('related_topics', [])
            else:  # figure, formula, etc.
                result['content'] = parsed.get('description', '')
                result['analysis'] = parsed.get('educational_value', '')
                result['related_topics'] = parsed.get('related_topics', [])
                result['exam_relevance'] = parsed.get('exam_relevance', '')
            
            return result
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini JSON response, using as plain text")
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'content': response_text,
                'analysis': 'Direct response (JSON parsing failed)',
                'confidence': region_info.get('confidence', 1.0)
            }


class SiliconCloudClient(BaseLLMClient):
    """SiliconCloud API client implementation"""
    
    def __init__(self, model: str = "THUDM/GLM-4.1V-9B-Thinking", api_key: Optional[str] = None):
        """Initialize SiliconCloud API client"""
        super().__init__(model, api_key)
        # 支持多个环境变量名，优先使用SILICONFLOW_API_KEY
        self.api_key = api_key or os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("SILICONCLOUD_API_KEY") or os.environ.get("SILICON_CLOUD_API_KEY")
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.client = self._setup_client()
    
    def _setup_client(self):
        """Setup SiliconCloud API client"""
        if not self.api_key:
            logger.warning("SILICON_CLOUD_API_KEY environment variable not set")
            return None
        
        logger.info("SiliconCloud API client initialized successfully")
        return self  # Using requests directly, so self is sufficient
    
    def is_available(self) -> bool:
        """Check if SiliconCloud API client is available"""
        return self.api_key is not None
    
    def extract_text(self, region_img: np.ndarray, region_info: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Extract text from region using SiliconCloud API"""
        if not self.is_available():
            logger.warning("SiliconCloud API client not initialized")
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'text': '',
                'confidence': 0.0
            }
        
        try:
            # Resize image if too large
            region_img_resized = self._resize_image_if_needed(region_img)
            img_bytes = self._convert_image_to_bytes(region_img_resized)
            
            # Base64 encode the image
            import base64
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            
            # Prepare payload
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}  
                        ]
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.1
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            logger.info(f"Requesting SiliconCloud extract_text (model={self.model})")
            # Increase timeout and add retry mechanism
            max_retries = 3
            retry_count = 0
            response = None
            
            while retry_count < max_retries:
                try:
                    response = requests.post(self.base_url, json=payload, headers=headers, timeout=60)  # Increased timeout to 60 seconds
                    break  # Exit loop if request succeeds
                except requests.exceptions.Timeout:
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"SiliconCloud request timeout, retrying ({retry_count}/{max_retries})...")
                        time.sleep(2)  # Wait before retrying
                    else:
                        logger.error("SiliconCloud request timed out after multiple attempts")
                        raise
                except Exception:
                    raise
            
            del region_img_resized, img_bytes, base64_image
            gc.collect()
            
            response_json = response.json()
            
            # Handle rate limits
            if response.status_code == 429:
                logger.error("SiliconCloud rate limit exceeded")
                return {
                    'type': region_info['type'],
                    'coords': region_info['coords'],
                    'text': '[RATE_LIMIT_EXCEEDED]',
                    'confidence': 0.0,
                    'error': 'siliconcloud_rate_limit',
                    'error_message': 'SiliconCloud API rate limit exceeded'
                }
            
            # Handle other errors
            if response.status_code != 200:
                error_msg = response_json.get('error', {}).get('message', str(response.status_code))
                logger.error(f"SiliconCloud API error: {error_msg}")
                return {
                    'type': region_info['type'],
                    'coords': region_info['coords'],
                    'text': '[SILICONCLOUD_EXTRACTION_FAILED]',
                    'confidence': 0.0,
                    'error': 'siliconcloud_api_error',
                    'error_message': error_msg
                }
            
            text = response_json['choices'][0]['message']['content'].strip()
            
            result = {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'text': text,
                'confidence': region_info.get('confidence', 1.0)
            }
            
            del response, response_json
            gc.collect()
            
            return result
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"SiliconCloud text extraction error: {e}")
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'text': '[SILICONCLOUD_EXTRACTION_FAILED]',
                'confidence': 0.0,
                'error': 'siliconcloud_exception',
                'error_message': error_str
            }
    
    def process_special_region(self, region_img: np.ndarray, region_info: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Process special regions (tables, figures) with SiliconCloud API"""
        if not self.is_available():
            logger.warning("SiliconCloud API client not initialized")
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'content': 'SiliconCloud API not available',
                'analysis': 'Client not initialized',
                'confidence': 0.0
            }
        
        try:
            # Use extract_text method with special prompt
            result = self.extract_text(region_img, region_info, prompt)
            
            # Convert result format to match expected output
            formatted_result = {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'content': result.get('text', ''),
                'analysis': 'Processed by SiliconCloud',
                'confidence': result.get('confidence', 0.0)
            }
            
            # Add error information if present
            if 'error' in result:
                formatted_result['error'] = result['error']
                formatted_result['error_message'] = result.get('error_message', '')
            
            return formatted_result
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"SiliconCloud special region processing error: {e}")
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'content': '[SILICONCLOUD_PROCESSING_FAILED]',
                'analysis': f'Processing failed: {error_str}',
                'confidence': 0.0,
                'error': 'siliconcloud_exception',
                'error_message': error_str
            }
    
    def correct_text(self, text: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Correct OCR text using SiliconCloud API"""
        if not self.is_available() or not text:
            return {"corrected_text": text, "confidence": 0.0}
        
        try:
            # Prepare payload with text content
            full_prompt = f"{system_prompt}\n\n{user_prompt}\n\n{text}"
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt}
                        ]
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.1
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            logger.info(f"Requesting SiliconCloud correct_text (model={self.model})")
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
            
            response_json = response.json()
            
            # Handle rate limits
            if response.status_code == 429:
                logger.error("SiliconCloud rate limit exceeded during text correction")
                return {"corrected_text": text, "confidence": 0.0, "error": "rate_limit"}
            
            # Handle other errors
            if response.status_code != 200:
                error_msg = response_json.get('error', {}).get('message', str(response.status_code))
                logger.error(f"SiliconCloud API error during text correction: {error_msg}")
                return {"corrected_text": text, "confidence": 0.0, "error": "correction_failed"}
            
            corrected_text = response_json['choices'][0]['message']['content'].strip()
            
            sm = difflib.SequenceMatcher(None, text, corrected_text)
            confidence = sm.ratio()

            return {
                "corrected_text": corrected_text,
                "confidence": confidence
            }
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"SiliconCloud text correction error: {e}")
            return {"corrected_text": text, "confidence": 0.0, "error": "correction_failed"}


class LLMClientManager:
    """Manager for multiple LLM clients, providing fallback functionality"""
    
    def __init__(self):
        """Initialize client manager"""
        self.clients = []
        self.current_client_index = 0
        self.last_used_client = None
    
    def register_client(self, client: BaseLLMClient):
        """Register a client to the manager"""
        if client.is_available():
            self.clients.append(client)
            logger.info(f"Registered LLM client: {client.__class__.__name__} (model={client.model})\n")
        else:
            logger.warning(f"Client not available, skipping: {client.__class__.__name__}")
    
    def add_client(self, client: BaseLLMClient):
        """Add a client to the manager (backwards compatibility)"""
        self.register_client(client)
    
    def get_current_client(self) -> Optional[BaseLLMClient]:
        """Get the current active client"""
        if not self.clients:
            return None
        return self.clients[self.current_client_index]
    
    def get_available_clients(self) -> list:
        """Get all available clients"""
        return [client for client in self.clients if client.is_available()]
    
    def switch_to_next_client(self) -> bool:
        """Switch to the next available client"""
        if len(self.clients) <= 1:
            return False
        
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        logger.info(f"Switched to client: {self.clients[self.current_client_index].__class__.__name__}")
        return True
    
    def handle_rate_limit(self) -> bool:
        """Handle rate limit by switching to next client or waiting"""
        # First try to switch to another client
        if self.switch_to_next_client():
            return True
        
        # If no other clients, wait
        logger.warning("No other clients available. Waiting before retrying...")
        time.sleep(5)  # Simple fixed wait, could be improved with exponential backoff
        return True
    
    def extract_text(self, region_img: np.ndarray, region_info: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Extract text from region using available LLM API with fallback"""
        attempts = 0
        max_attempts = len(self.clients) * 2  # Try each client twice
        
        while attempts < max_attempts:
            client = self.get_current_client()
            if not client:
                logger.error("No available LLM clients")
                return {
                    'type': region_info['type'],
                    'coords': region_info['coords'],
                    'text': '[NO_AVAILABLE_CLIENTS]',
                    'confidence': 0.0,
                    'error': 'no_clients'
                }
            
            try:
                result = client.extract_text(region_img, region_info, prompt)
                self.last_used_client = client
                
                # Check for rate limit or other recoverable errors
                if 'error' in result:
                    if result['error'] in ['gemini_rate_limit', 'rate_limit']:
                        logger.warning(f"Rate limit hit on {client.__class__.__name__}. Attempting fallback...")
                        
                        # 尝试从错误消息中提取建议的等待时间
                        wait_time = self._extract_wait_time_from_error(result.get('error_message', ''))
                        
                        # 查找SiliconCloudClient（THUDM/GLM-4.1V-9B-Thinking）
                        silicon_client = None
                        for c in self.clients:
                            if isinstance(c, SiliconCloudClient) and c.model == "THUDM/GLM-4.1V-9B-Thinking":
                                silicon_client = c
                                break
                        
                        if silicon_client and wait_time > 0:
                            logger.info(f"Gemini rate limited, will try THUDM/GLM-4.1V-9B-Thinking while waiting {wait_time}s")
                            
                            # 使用线程异步调用GLM-4.1V-9B-Thinking
                            silicon_result = None
                            silicon_exception = None
                            result_lock = threading.Lock()
                            
                            def silicon_worker():
                                nonlocal silicon_result, silicon_exception
                                try:
                                    with result_lock:
                                        silicon_result = silicon_client.extract_text(region_img, region_info, prompt)
                                except Exception as e:
                                    silicon_exception = e
                                    logger.error(f"SiliconCloud extraction failed: {e}")
                            
                            thread = threading.Thread(target=silicon_worker)
                            thread.daemon = True
                            thread.start()
                            
                            # 等待指定时间，但最多等待wait_time秒
                            start_time = time.time()
                            while time.time() - start_time < wait_time:
                                if thread.is_alive():
                                    time.sleep(0.1)
                                else:
                                    break
                            
                            # 检查是否有结果
                            with result_lock:
                                if silicon_result and 'error' not in silicon_result:
                                    logger.info(f"Successfully got result from THUDM/GLM-4.1V-9B-Thinking")
                                    return silicon_result
                        
                        # 如果没有SiliconCloudClient或者在等待时间内没有得到有效响应，就等待然后重试Gemini
                        logger.info(f"Waiting {wait_time}s before retrying Gemini")
                        time.sleep(wait_time)
                        attempts += 1
                        continue
                    
                return result
                
            except Exception as e:
                logger.error(f"Error with {client.__class__.__name__}: {e}")
                self.switch_to_next_client()
                attempts += 1
        
        logger.error("All LLM clients failed")
        return {
            'type': region_info['type'],
            'coords': region_info['coords'],
            'text': '[ALL_CLIENTS_FAILED]',
            'confidence': 0.0,
            'error': 'all_clients_failed'
        }
        
    def _extract_wait_time_from_error(self, error_message: str) -> float:
        """从Gemini的错误消息中提取建议的等待时间"""
        # 尝试匹配形如 "Please retry in 13.916996282s." 或 "retryDelay': '13s" 的模式
        try:
            # 匹配第一种模式
            match = re.search(r'Please retry in (\d+(\.\d+)?)s', error_message)
            if match:
                return float(match.group(1))
            
            # 匹配第二种模式
            match = re.search(r"retryDelay': '?(\d+)s'?", error_message)
            if match:
                return float(match.group(1))
            
            # 匹配数字加秒的模式
            match = re.search(r'(\d+(\.\d+)?)\s*秒', error_message)
            if match:
                return float(match.group(1))
                
            # 默认等待5秒
            return 5.0
        except:
            return 5.0
    
    def generate_summary(self, content: str, prompt: str) -> str:
        """Generate summary for content using available LLM API with fallback"""
        attempts = 0
        max_attempts = len(self.clients) * 2  # Try each client twice
        
        while attempts < max_attempts:
            client = self.get_current_client()
            if not client:
                logger.error("No available LLM clients for summarization")
                return "[NO_AVAILABLE_CLIENTS_FOR_SUMMARIZATION]"
            
            try:
                # 构建直接生成摘要的提示
                if "keywords" in prompt.lower():
                    # 关键词提取提示
                    full_prompt = f"{prompt}\n\n{content}\n\n请严格按照要求格式输出，只返回英文关键词列表，不要包含其他解释性文字。"
                else:
                    # 摘要生成提示
                    full_prompt = f"{prompt}\n\n{content}\n\n请确保摘要简短，用英文概括主要主题或结论，不要换行，控制在300字以内。"
                
                # 直接使用文本接口，不再使用需要图像输入的extract_text方法
                system_prompt = "You are an expert in document summarization."
                
                # 优先使用correct_text方法，正确传入content作为text参数
                if hasattr(client, 'correct_text'):
                    result = client.correct_text(content, system_prompt, full_prompt)
                    summary = result.get('corrected_text', '[SUMMARIZATION_FAILED]')
                else:
                    summary = "[NO_SUITABLE_METHOD_FOR_SUMMARIZATION]"
                
                self.last_used_client = client
                
                # 清理摘要内容，确保符合要求
                # 移除可能的换行符
                summary = summary.replace('\n', ' ')
                # 如果是关键词提取，确保返回的是逗号分隔的列表
                if "keywords" in prompt.lower() and summary and not summary.startswith('['):
                    # 检查是否已经是逗号分隔的格式
                    if ',' not in summary and len(summary.split()) > 1:
                        # 如果不是逗号分隔，尝试转换
                        summary = ', '.join(summary.split())
                
                return summary
                
            except Exception as e:
                logger.error(f"Error with {client.__class__.__name__}: {e}")
                self.switch_to_next_client()
                attempts += 1
        
        logger.error("All LLM clients failed for summarization")
        return "[ALL_CLIENTS_FAILED_FOR_SUMMARIZATION]"


# Legacy GeminiClient class to maintain backwards compatibility
class GeminiClient(GeminiLLMClient):
    """Legacy GeminiClient class for backwards compatibility"""
    pass