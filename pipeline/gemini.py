"""
Google Gemini VLM API client for advanced OCR text extraction and processing.
Handles text extraction, special content analysis, and text correction using Gemini Vision Language Models.
"""

import difflib
import gc
import io
import logging
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
from google import genai
from google.genai import types
from PIL import Image

logger = logging.getLogger(__name__)


class GeminiClient:
    """Google Gemini VLM API client for OCR text processing"""
    
    def __init__(self, gemini_model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize Gemini API client
        
        Args:
            gemini_model: Gemini model to use
            api_key: Gemini API key (if not provided, reads from environment)
        """
        self.gemini_model = gemini_model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = self._setup_gemini_api()
    
    def _setup_gemini_api(self) -> Optional[genai.Client]:
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
        """
        Extract text from region using Gemini API
        
        Args:
            region_img: Image region as numpy array
            region_info: Region metadata including type and coordinates
            prompt: Prompt for text extraction
            
        Returns:
            Dictionary containing extracted text and metadata
        """
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
            h, w = region_img.shape[:2]
            max_dim = 1024
            
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                region_img_resized = cv2.resize(region_img, (new_w, new_h))
            else:
                region_img_resized = region_img
            
            pil_image = Image.fromarray(cv2.cvtColor(region_img_resized, cv2.COLOR_BGR2RGB))
            
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
            img_bytes = img_byte_arr.getvalue()

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

            logger.info(f"Requesting Gemini extract_text (model={self.gemini_model})")
            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config=generate_content_config,
            )
            
            del pil_image, img_byte_arr, img_bytes, region_img_resized
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
                logger.error("Rate limit exceeded. Please wait before retrying or check your API quota.")
                return {
                    'type': region_info['type'],
                    'coords': region_info['coords'],
                    'text': '[RATE_LIMIT_EXCEEDED]',
                    'confidence': 0.0,
                    'error': 'gemini_rate_limit',
                    'error_message': 'Gemini API rate limit exceeded'
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
        """
        Process special regions (tables, figures) with Gemini API
        
        Args:
            region_img: Image region as numpy array
            region_info: Region metadata including type and coordinates
            prompt: Prompt for special content analysis
            
        Returns:
            Dictionary containing processed content and metadata
        """
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
            h, w = region_img.shape[:2]
            max_dim = 1024
            
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                region_img_resized = cv2.resize(region_img, (new_w, new_h))
            else:
                region_img_resized = region_img
            
            pil_image = Image.fromarray(cv2.cvtColor(region_img_resized, cv2.COLOR_BGR2RGB))
            
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
            img_bytes = img_byte_arr.getvalue()

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

            logger.info(f"Requesting Gemini process_special_region (model={self.gemini_model})")
            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config=generate_content_config,
            )
            
            del pil_image, img_byte_arr, img_bytes, region_img_resized
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
        """
        Correct OCR text using Gemini API
        
        Args:
            text: Text to correct
            system_prompt: System instruction prompt
            user_prompt: User prompt with text formatting
            
        Returns:
            Dictionary containing corrected text and confidence
        """
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

            logger.info(f"Requesting Gemini correct_text (model={self.gemini_model})")
            response = self.client.models.generate_content(
                model=self.gemini_model,
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
                return "[TEXT_CORRECTION_RATE_LIMIT_EXCEEDED]"
            
            # Handle service unavailable errors
            elif "503" in error_str or "UNAVAILABLE" in error_str:
                logger.error("Service unavailable during text correction")
                return "[TEXT_CORRECTION_SERVICE_UNAVAILABLE]"
            
            # For other errors, return original text with error indicator
            else:
                logger.error("Text correction failed with other error")
                return f"[TEXT_CORRECTION_FAILED]: {text}"
    
    def _parse_gemini_response(self, response_text: str, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Gemini response for special regions"""
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
            logger.warning("Failed to parse Gemini JSON response, using as plain text")
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'content': response_text,
                'analysis': 'Direct response (JSON parsing failed)',
                'confidence': region_info.get('confidence', 1.0)
            }
    
    def reload_client(self, api_key: Optional[str] = None) -> bool:
        """
        Reload the Gemini API client (useful after API key updates)
        
        Args:
            api_key: New API key to use (optional)
            
        Returns:
            True if client was successfully reloaded
        """
        if api_key:
            self.api_key = api_key
        
        self.client = self._setup_gemini_api()
        return self.is_available()