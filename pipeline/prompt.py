"""
Prompt management system for VLM-based document processing.
Handles loading, selection, and customization of prompts for different document regions.
"""

import glob
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompt templates for different document region types"""
    
    def __init__(self, model: str = "gemini-2.5-flash", backend: str = "gemini"):
        """
        Initialize prompt manager
        
        Args:
            model: Model name being used
            backend: Backend type (gemini, gpt, etc.)
        """
        self.model = model
        self.backend = backend.lower()
        self.prompts_dir = self._find_best_prompts_dir()
        self.prompts: Dict[str, Dict[str, str]] = {}
        
        if self.prompts_dir:
            self._load_prompts()
        else:
            logger.warning("No prompts directory found, will use fallback prompts")
    
    def _find_best_prompts_dir(self) -> Optional[Path]:
        """Find the most appropriate prompts directory based on model and backend"""
        # First try exact model match
        possible_dirs = [
            Path(__file__).parent.parent / "prompts" / f"{self.backend}_{self.model}",
            Path(__file__).parent.parent / "prompts" / f"{self.backend}_{self.model.split('-')[0]}",
            Path(__file__).parent.parent / "prompts" / self.backend,
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists() and dir_path.is_dir():
                logger.info(f"Found prompts directory: {dir_path}")
                return dir_path
        
        # Check default location
        default_dir = Path(__file__).parent.parent / "prompts"
        if default_dir.exists() and default_dir.is_dir():
            logger.info(f"Using default prompts directory: {default_dir}")
            return default_dir
        
        return None
    
    def _load_prompts(self) -> None:
        """Load prompts from YAML files in the prompts directory"""
        if not self.prompts_dir:
            return
        
        try:
            prompt_files = glob.glob(str(self.prompts_dir / "*.yaml")) + \
                           glob.glob(str(self.prompts_dir / "*.yml"))
            
            for prompt_file in prompt_files:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_data = yaml.safe_load(f)
                    if prompt_data:
                        prompt_name = Path(prompt_file).stem
                        self.prompts[prompt_name] = prompt_data
                        logger.debug(f"Loaded prompt: {prompt_name}")
            
            if not self.prompts:
                logger.warning(f"No prompts found in directory: {self.prompts_dir}")
            else:
                logger.info(f"Loaded {len(self.prompts)} prompt templates")
                
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
    
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Get a prompt by name with optional template variables
        
        Args:
            prompt_name: Name of the prompt to retrieve
            **kwargs: Template variables to substitute
            
        Returns:
            Formatted prompt string
        """
        # Check if prompt exists
        if prompt_name in self.prompts:
            prompt_template = self.prompts[prompt_name].get('prompt', '')
            if prompt_template:
                try:
                    return prompt_template.format(**kwargs)
                except KeyError as e:
                    logger.warning(f"Missing template variable in prompt '{prompt_name}': {e}")
                    # Return original template without formatting
                    return prompt_template
                except Exception as e:
                    logger.error(f"Error formatting prompt '{prompt_name}': {e}")
                    return prompt_template
        
        # Fall back to a default prompt
        logger.warning(f"Prompt '{prompt_name}' not found, using fallback")
        return self._get_fallback_prompt(prompt_name)
    
    def _get_fallback_prompt(self, prompt_name: str) -> str:
        """Get a fallback prompt when the requested prompt is not found"""
        fallback_prompts = {
            'text_extraction': "Extract all text from this document region. Preserve formatting as much as possible.",
            'table_extraction': "Analyze this table image and convert it to Markdown format. Include a brief summary of the data.",
            'figure_analysis': "Describe this figure/image in detail. Highlight key elements and their significance.",
            'formula_analysis': "Recognize and describe this mathematical formula. Explain what it represents.",
            'page_summary': "Summarize the content of this page in clear, concise language.",
            'document_structure': "Analyze the structure of this document and identify key sections.",
        }
        
        return fallback_prompts.get(prompt_name, 
                                  f"Please analyze this document content and provide a detailed response.")
    
    # Mapping of region types to analysis types
    region_type_mapping = {
        'text': 'text_extraction',
        'heading': 'text_extraction',
        'title': 'text_extraction',
        'subtitle': 'text_extraction',
        'paragraph': 'text_extraction',
        'caption': 'text_extraction',
        'list': 'text_extraction',
        'table': 'table_extraction',
        'figure': 'figure_analysis',
        'formula': 'formula_analysis',
        'image': 'figure_analysis',
        'chart': 'figure_analysis',
        'diagram': 'figure_analysis'
    }
    
    def get_prompt_for_region_type(self, region_type: str, **kwargs) -> str:
        """
        Get the appropriate prompt for a specific region type
        
        Args:
            region_type: Type of document region
            **kwargs: Template variables to substitute
            
        Returns:
            Formatted prompt string for the region type
        """
        analysis_type = self.region_type_mapping.get(region_type.lower(), 'text_extraction')
        return self.get_prompt(analysis_type, **kwargs)
    
    def get_gemini_prompt_for_region_type(self, region_type: str, **kwargs) -> str:
        """
        DEPRECATED: Use get_prompt_for_region_type instead
        
        Get Gemini-specific prompt for a region type
        """
        logger.warning("get_gemini_prompt_for_region_type is deprecated, use get_prompt_for_region_type instead")
        return self.get_prompt_for_region_type(region_type, **kwargs)
    
    def reload_prompts(self) -> None:
        """Reload prompts from disk (useful for development)"""
        logger.info("Reloading prompts from disk")
        self.prompts = {}
        self.prompts_dir = self._find_best_prompts_dir()
        self._load_prompts()