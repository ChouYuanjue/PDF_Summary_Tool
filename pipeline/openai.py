"""
OpenAI VLM API client for advanced OCR text extraction and processing.
Compatible with OpenAI and OpenRouter Vision Language Model APIs.
Handles text extraction, special content analysis, and text correction.
"""

import difflib
import gc
import io
import logging
import os
import base64
import re
from typing import Any, Dict, Optional

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI VLM API client for OCR text processing"""
    
    def __init__(self, model: str = "gpt-5-nano", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize OpenAI API client
        
        Args:
            model: Model to use (can be OpenRouter format like 'openai/gpt-4')
            api_key: API key (if not provided, reads from environment)
            base_url: Base URL for API (for OpenRouter or custom endpoints)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        
        # Set default base URL for OpenRouter if using openrouter models
        if not self.base_url and ("/" in self.model or "anthropic" in self.model.lower() or "google" in self.model.lower()):
            self.base_url = "https://openrouter.ai/api/v1"
            # Try OpenRouter API key if OpenAI key not available
            if not self.api_key:
                self.api_key = os.environ.get("OPENROUTER_API_KEY")
        
        self.client = self._setup_openai_client()
    
    def _setup_openai_client(self) -> Optional[OpenAI]:
        """Setup OpenAI API client"""
        try:
            if not self.api_key:
                logger.warning("OPENAI_API_KEY or OPENROUTER_API_KEY environment variable not set")
                return None
            
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            client = OpenAI(**client_kwargs)
            logger.info(f"OpenAI API client initialized successfully (base_url: {self.base_url or 'default'})")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI API client: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if OpenAI API client is available"""
        return self.client is not None
        
    def _clean_response_text(self, text: str) -> str:
        """Clean response text by removing content wrapped in</think> and</think>"""
        # 匹配并移除被</think>和</think>包裹的内容
        cleaned_text = re.sub(r'\s*\-*\s*\{\{\|\s*(.+?)\s*\|\}\}\s*\-*', '', text, flags=re.DOTALL)
        
        return cleaned_text.strip()
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 for OpenAI API"""
        # Resize image if too large
        h, w = image.shape[:2]
        max_dim = 1024
        
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_resized = cv2.resize(image, (new_w, new_h))
        else:
            image_resized = image
        
        pil_image = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
        
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
        img_bytes = img_byte_arr.getvalue()
        
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def extract_text(self, region_img: np.ndarray, region_info: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """
        Extract text from region using OpenAI API
        
        Args:
            region_img: Image region as numpy array
            region_info: Region metadata including type and coordinates
            prompt: Prompt for text extraction
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        if not self.is_available():
            logger.warning(f"OpenAI API client not initialized (model={self.model}, base_url={self.base_url or 'default'})")
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'text': '',
                'confidence': 0.0
            }
        
        try:
            base64_image = self._encode_image(region_img)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            logger.info(f"Requesting OpenAI extract_text (model={self.model}, base_url={self.base_url or 'default'})")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.1
            )
            
            text = response.choices[0].message.content.strip()
            
            result = {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'text': text,
                'confidence': region_info.get('confidence', 1.0)
            }
            
            # Clean up
            del base64_image
            gc.collect()
            
            return result
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"OpenAI text extraction error: {e}")
            
            # Handle rate limit errors specifically
            if "429" in error_str or "rate_limit" in error_str.lower():
                logger.error("Rate limit exceeded. Please wait before retrying or check your API quota.")
                return {
                    'type': region_info['type'],
                    'coords': region_info['coords'],
                    'text': '[RATE_LIMIT_EXCEEDED]',
                    'confidence': 0.0,
                    'error': 'openai_rate_limit',
                    'error_message': 'OpenAI API rate limit exceeded'
                }
            
            # Handle other OpenAI API errors
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'text': '[OPENAI_EXTRACTION_FAILED]',
                'confidence': 0.0,
                'error': 'openai_api_error',
                'error_message': str(e)
            }
    
    def process_special_region(self, region_img: np.ndarray, region_info: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """
        Process special regions (tables, figures) with OpenAI API
        
        Args:
            region_img: Image region as numpy array
            region_info: Region metadata including type and coordinates
            prompt: Prompt for special content analysis
            
        Returns:
            Dictionary containing processed content and metadata
        """
        if not self.is_available():
            logger.warning(f"OpenAI API client not initialized (model={self.model}, base_url={self.base_url or 'default'})")
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'content': 'OpenAI API not available',
                'analysis': 'Client not initialized',
                'confidence': 0.0
            }
        
        try:
            base64_image = self._encode_image(region_img)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            logger.info(f"Requesting OpenAI process_special_region (model={self.model}, base_url={self.base_url or 'default'})")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=3000,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            # Clean response text
            cleaned_text = self._clean_response_text(response_text)
            parsed_result = self._parse_openai_response(cleaned_text, region_info)
            
            # Clean up
            del base64_image
            gc.collect()
            
            return parsed_result
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"OpenAI special region processing error: {e}")
            
            # Handle rate limit errors
            if "429" in error_str or "rate_limit" in error_str.lower():
                return {
                    'type': region_info['type'],
                    'coords': region_info['coords'],
                    'content': '[RATE_LIMIT_EXCEEDED]',
                    'analysis': 'Rate limit exceeded',
                    'confidence': 0.0,
                    'error': 'openai_rate_limit',
                    'error_message': 'OpenAI API rate limit exceeded'
                }
            
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'content': '[OPENAI_PROCESSING_FAILED]',
                'analysis': f'Processing failed: {str(e)}',
                'confidence': 0.0,
                'error': 'openai_api_error',
                'error_message': str(e)
            }
    
    def correct_text(self, text: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Correct OCR text using OpenAI API"""
        if not self.is_available() or not text:
            return {"corrected_text": text, "confidence": 0.0}
        
        try:
            # 构建完整的提示，包含文本内容
            full_user_prompt = f"{user_prompt}\n\n{text}"
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": full_user_prompt
                }
            ]
            
            logger.info(f"Requesting OpenAI correct_text (model={self.model}, base_url={self.base_url or 'default'})")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=len(text.split()) * 3,  # Allow for expansion
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            # Clean response text
            cleaned_text = self._clean_response_text(response_text)
            
            sm = difflib.SequenceMatcher(None, text, cleaned_text)
            confidence = sm.ratio()
            
            return {
                "corrected_text": cleaned_text,
                "confidence": confidence
            }
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Text correction error: {e}")
            
            # Handle rate limit errors specifically
            if "429" in error_str or "rate_limit" in error_str.lower():
                logger.error("Rate limit exceeded during text correction")
                return {"corrected_text": text, "confidence": 0.0, "error": "rate_limit"}
            
            # Handle service unavailable errors
            elif "503" in error_str or "unavailable" in error_str.lower():
                logger.error("Service unavailable during text correction")
                return {"corrected_text": text, "confidence": 0.0, "error": "service_unavailable"}
            
            # For other errors, return original text with error indicator
            else:
                logger.error("Text correction failed with other error")
                return {"corrected_text": text, "confidence": 0.0, "error": "correction_failed"}
    
    def _parse_openai_response(self, response_text: str, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenAI response for special regions"""
        try:
            import json
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
            logger.warning("Failed to parse OpenAI JSON response, using as plain text")
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'content': response_text,
                'analysis': 'Direct response (JSON parsing failed)',
                'confidence': region_info.get('confidence', 1.0)
            }
    
    def reload_client(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> bool:
        """
        Reload the OpenAI API client (useful after API key updates)
        
        Args:
            api_key: New API key to use (optional)
            base_url: New base URL to use (optional)
            
        Returns:
            True if client was successfully reloaded
        """
        if api_key:
            self.api_key = api_key
        if base_url:
            self.base_url = base_url
        
        self.client = self._setup_openai_client()
        return self.is_available()