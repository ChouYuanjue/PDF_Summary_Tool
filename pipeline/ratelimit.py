"""
Rate limiting system for managing API calls to Gemini and other services.
Implements token bucket and leaky bucket algorithms for rate control.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class RateLimitManager:
    """Rate limit manager for controlling API call frequency"""
    # Singleton instance
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RateLimitManager, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize rate limit manager with default settings"""
        # Default rate limits per model (RPM, TPM, RPD)
        self.default_limits = {
            "gemini-2.5-flash": (60, 30000, 500),    # 60 RPM, 30K TPM, 500 RPD
            "gemini-2.5-pro": (30, 20000, 300),      # 30 RPM, 20K TPM, 300 RPD
            "gemini-1.5-pro": (20, 15000, 200),      # 20 RPM, 15K TPM, 200 RPD
            "default": (15, 10000, 150)              # 15 RPM, 10K TPM, 150 RPD
        }
        
        # Rate limit status tracking
        self.rate_limit_status: Dict[str, Dict] = {}
        
        # State file for persistence
        self.state_file = Path(__file__).parent.parent / "rate_limit_state.json"
        
        # Threading locks for thread safety
        self.status_lock = threading.Lock()
        
        # Load saved state if available
        self._load_state()
    
    def _load_state(self):
        """Load rate limit state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    saved_state = json.load(f)
                    
                    # Only load state that is not too old
                    current_time = time.time()
                    valid_state = {}
                    
                    for model, state in saved_state.items():
                        # Check if state is within the last hour
                        if current_time - state.get('last_update', 0) < 3600:
                            valid_state[model] = state
                    
                    self.rate_limit_status = valid_state
                    logger.info(f"Loaded rate limit state for {len(valid_state)} models")
        except Exception as e:
            logger.error(f"Failed to load rate limit state: {e}")
    
    def _save_state(self):
        """Save rate limit state to file"""
        try:
            current_time = time.time()
            
            # Update last_update timestamp for all models
            state_to_save = {}
            for model, state in self.rate_limit_status.items():
                state_copy = state.copy()
                state_copy['last_update'] = current_time
                state_to_save[model] = state_copy
            
            # Create directory if it doesn't exist
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save rate limit state: {e}")
    
    def _get_limits_for_model(self, model: str) -> Tuple[int, int, int]:
        """\Get rate limits for a specific model"""
        # Try exact match first
        if model in self.default_limits:
            return self.default_limits[model]
            
        # Try partial match
        for model_prefix, limits in self.default_limits.items():
            if model.startswith(model_prefix) and model_prefix != "default":
                return limits
                
        # Fall back to default
        return self.default_limits["default"]
    
    def _get_or_create_model_state(self, model: str) -> Dict:
        """Get or create rate limit state for a model"""
        with self.status_lock:
            if model not in self.rate_limit_status:
                self.rate_limit_status[model] = {
                    "requests_last_minute": [],  # Timestamps of recent requests
                    "tokens_last_minute": [],    # (timestamp, token_count) pairs
                    "requests_today": 0,         # Total requests today
                    "last_reset_day": self._get_current_day()
                }
            return self.rate_limit_status[model]
    
    def _get_current_day(self) -> int:
        """Get current day as a integer (days since epoch)"""
        return int(time.time() // 86400)
    
    def _cleanup_old_records(self, model_state: Dict):
        """Clean up old rate limit records"""
        current_time = time.time()
        
        # Clean up requests older than 1 minute
        model_state["requests_last_minute"] = [
            ts for ts in model_state["requests_last_minute"] 
            if current_time - ts < 60
        ]
        
        # Clean up tokens older than 1 minute
        model_state["tokens_last_minute"] = [
            (ts, tokens) for ts, tokens in model_state["tokens_last_minute"] 
            if current_time - ts < 60
        ]
        
        # Reset daily count if new day
        current_day = self._get_current_day()
        if model_state["last_reset_day"] != current_day:
            model_state["requests_today"] = 0
            model_state["last_reset_day"] = current_day
    
    def _calculate_wait_time(self, model: str, token_count: int = 100) -> float:
        """
        Calculate how long to wait before making the next request
        
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        current_time = time.time()
        rpm, tpm, rpd = self._get_limits_for_model(model)
        model_state = self._get_or_create_model_state(model)
        
        # Clean up old records
        self._cleanup_old_records(model_state)
        
        # Check daily limit
        if model_state["requests_today"] >= rpd:
            # Wait until next day
            seconds_until_midnight = (self._get_current_day() + 1) * 86400 - current_time
            logger.warning(f"Daily rate limit exceeded for {model}. Waiting {seconds_until_midnight:.1f}s")
            return seconds_until_midnight
        
        # Check RPM
        requests_in_window = len(model_state["requests_last_minute"])
        if requests_in_window >= rpm:
            # Calculate how long to wait until the oldest request falls out of the window
            oldest_request = model_state["requests_last_minute"][0]
            wait_time_rpm = 60 - (current_time - oldest_request)
            if wait_time_rpm > 0:
                logger.debug(f"RPM limit reached for {model}. Waiting {wait_time_rpm:.1f}s")
                return wait_time_rpm
        
        # Check TPM (approximation - assuming each token is 4 characters)
        total_tokens_in_window = sum(tokens for _, tokens in model_state["tokens_last_minute"])
        if total_tokens_in_window + token_count > tpm:
            # Calculate how much time until oldest token records fall out of window
            # This is a simplified calculation
            oldest_token_time = model_state["tokens_last_minute"][0][0] if model_state["tokens_last_minute"] else 0
            wait_time_tpm = 60 - (current_time - oldest_token_time)
            if wait_time_tpm > 0:
                logger.debug(f"TPM limit approaching for {model}. Waiting {wait_time_tpm:.1f}s")
                return wait_time_tpm
        
        # No wait needed
        return 0
    
    def wait_if_needed(self, model: str, token_count: int = 100) -> bool:
        """
        Wait if rate limits require it before making a request
        
        Args:
            model: Model name
            token_count: Estimated token count for the request
            
        Returns:
            True if we can proceed with the request, False if rate limits are exceeded
        """
        wait_time = self._calculate_wait_time(model, token_count)
        
        if wait_time > 0:
            # For very long waits (over 30 minutes), we might want to fail instead
            if wait_time > 1800:
                logger.error(f"Rate limit would require waiting over 30 minutes. Aborting request.")
                return False
            
            # Implement exponential backoff if needed
            time.sleep(wait_time)
            
            # Double-check after waiting
            return self.wait_if_needed(model, token_count)
            
        return True
    
    def record_request(self, model: str, token_count: int = 100):
        """
        Record a request against the rate limits
        
        Args:
            model: Model name
            token_count: Estimated token count for the request
        """
        current_time = time.time()
        model_state = self._get_or_create_model_state(model)
        
        with self.status_lock:
            # Record request time
            model_state["requests_last_minute"].append(current_time)
            
            # Record token count
            model_state["tokens_last_minute"].append((current_time, token_count))
            
            # Increment daily count
            model_state["requests_today"] += 1
        
        # Clean up old records
        self._cleanup_old_records(model_state)
        
        # Save state periodically (but not too often)
        if int(current_time) % 10 == 0:  # Every ~10 seconds
            self._save_state()
    
    def get_current_state(self, model: str) -> Dict:
        """
        Get current rate limit state for a model
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with current rate limit state
        """
        rpm, tpm, rpd = self._get_limits_for_model(model)
        model_state = self._get_or_create_model_state(model)
        self._cleanup_old_records(model_state)
        
        current_time = time.time()
        
        # Calculate token usage
        total_tokens_in_window = sum(tokens for _, tokens in model_state["tokens_last_minute"])
        
        return {
            "model": model,
            "limits": {
                "rpm": rpm,
                "tpm": tpm,
                "rpd": rpd
            },
            "current_usage": {
                "rpm": len(model_state["requests_last_minute"]),
                "tpm": total_tokens_in_window,
                "rpd": model_state["requests_today"]
            },
            "timestamp": current_time,
            "estimated_wait_time": self._calculate_wait_time(model)
        }
    
    def update_limits(self, model: str, rpm: Optional[int] = None, 
                      tpm: Optional[int] = None, rpd: Optional[int] = None):
        """
        Update rate limits for a specific model
        
        Args:
            model: Model name
            rpm: New RPM limit (optional)
            tpm: New TPM limit (optional)
            rpd: New RPD limit (optional)
        """
        if model not in self.default_limits:
            # Use current default as base
            current_rpm, current_tpm, current_rpd = self._get_limits_for_model(model)
            self.default_limits[model] = (
                rpm if rpm is not None else current_rpm,
                tpm if tpm is not None else current_tpm,
                rpd if rpd is not None else current_rpd
            )
        else:
            # Update existing limits
            current_rpm, current_tpm, current_rpd = self.default_limits[model]
            self.default_limits[model] = (
                rpm if rpm is not None else current_rpm,
                tpm if tpm is not None else current_tpm,
                rpd if rpd is not None else current_rpd
            )
            
        logger.info(f"Updated rate limits for {model}: RPM={self.default_limits[model][0]}, "
                   f"TPM={self.default_limits[model][1]}, RPD={self.default_limits[model][2]}")