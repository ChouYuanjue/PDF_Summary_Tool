import os
import argparse
import hashlib
import json
import gc
import logging
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any

import cv2
import numpy as np
import fitz  # PyMuPDF，作为不需要poppler的备选方案
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdf_to_md.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 导入本地实现的模型和工具类
from models.doclayout_yolo import DocLayoutYOLO
from pipeline.llm_client import GeminiLLMClient, SiliconCloudClient, LLMClientManager
from pipeline.openai import OpenAIClient
from pipeline.prompt import PromptManager

class PDFProcessor:
    """优化的PDF处理类，集成布局检测和区域特定处理"""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = 'google/gemini-2.0-flash-001',
                 confidence_threshold: float = 0.5,
                 use_cache: bool = True,
                 cache_dir: str = ".cache",
                 temp_dir: str = ".tmp",
                 use_full_page: bool = False):
        """初始化PDF处理器"""
        # 优先使用OPENROUTER_API_KEY
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.use_cache = use_cache
        
        # 转换路径为Path对象
        self.cache_dir = Path(cache_dir)
        self.temp_dir = Path(temp_dir)
        
        # 创建必要的目录
        self._setup_directories()
        
        # 是否直接处理整页图片而不分块
        self.use_full_page = use_full_page
        
        # 初始化组件
        if model.startswith('gpt-'):
            self.prompt_manager = PromptManager(model=model, backend="openai")
        else:
            self.prompt_manager = PromptManager(model=model, backend="gemini")
        self.layout_model = self._setup_layout_model()
        
        # 初始化LLM客户端管理器，支持多模型切换
        self.llm_manager = LLMClientManager()
        
        # 直接使用OpenRouter API
        self.openai_client = None
        if self.api_key:
            # 确保使用OpenRouter的base_url
            base_url = "https://openrouter.ai/api/v1"
            # 创建OpenAIClient实例，明确指定使用OpenRouter
            openai_client = OpenAIClient(model=self.model, api_key=self.api_key, base_url=base_url)
            self.llm_manager.register_client(openai_client)
            self.openai_client = openai_client
        
        # 添加THUDM/GLM-4.1V-9B-Thinking作为备选（当主要客户端遇到频率限制时使用）
        siliconcloud_api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("SILICONCLOUD_API_KEY")
        if siliconcloud_api_key:
            siliconcloud_client = SiliconCloudClient(model="THUDM/GLM-4.1V-9B-Thinking", api_key=siliconcloud_api_key)
            self.llm_manager.register_client(siliconcloud_client)
            
            # 如果主要客户端不可用，将GLM-4.1V-9B-Thinking设为当前客户端
            if not self.openai_client and not hasattr(self, 'gemini_client'):
                self.gemini_client = siliconcloud_client
    
    def _setup_directories(self) -> None:
        """创建必要的目录"""
        for directory in [self.cache_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_layout_model(self) -> DocLayoutYOLO:
        """设置DocLayout-YOLO模型"""
        try:
            model = DocLayoutYOLO()
            logger.info("DocLayout-YOLO模型加载成功")
            return model
        except Exception as e:
            logger.warning(f"DocLayout-YOLO模型加载失败，将使用简单模式: {str(e)}")
            return None
    

    def _calculate_image_hash(self, image: np.ndarray) -> str:
        """计算图像哈希值用于缓存"""
        small_img = cv2.resize(image, (32, 32))
        _, buffer = cv2.imencode('.jpg', small_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        image_hash = hashlib.md5(buffer).hexdigest()
        del small_img, buffer
        return image_hash
    
    def _get_cached_result(self, image_hash: str, cache_type: str) -> Optional[Dict[str, Any]]:
        """获取缓存的结果"""
        if not self.use_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_type}_{image_hash}.json"
        
        if cache_file.exists():
            try:
                with cache_file.open('r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                logger.debug(f"缓存命中: {cache_type}: {image_hash}")
                return cached_data
            except Exception as e:
                logger.warning(f"加载缓存文件失败 {cache_file}: {e}")
        
        return None
    
    def _save_to_cache(self, image_hash: str, cache_type: str, result: Dict[str, Any]) -> None:
        """将结果保存到缓存"""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_type}_{image_hash}.json"
        
        try:
            # 保存缓存数据时移除坐标信息
            cache_data = {k: v for k, v in result.items() if k != 'coords'}
            
            with cache_file.open('w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"已缓存结果: {cache_type}: {image_hash}")
        except Exception as e:
            logger.warning(f"保存缓存文件失败 {cache_file}: {e}")
    
    def _crop_region(self, image: np.ndarray, region: Dict[str, Any]) -> np.ndarray:
        """从图像中裁剪区域"""
        coords = region['coords']
        x, y, w, h = coords  # 坐标格式为 [x, y, width, height]
        
        # 转换为 x1, y1, x2, y2
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # 添加小的边距
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        # 确保有效尺寸
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"无效的区域坐标: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return np.zeros((1, 1, 3), dtype=np.uint8)  # 返回最小图像
        
        return image[y1:y2, x1:x2]

    def _extract_text_from_region(self, region_img: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """从区域中提取文本"""
        image_hash = self._calculate_image_hash(region_img)
        cache_type = f'gemini_ocr_{region_info["type"]}'
        
        # 检查缓存
        cached_result = self._get_cached_result(image_hash, cache_type)
        if cached_result is not None:
            cached_result['coords'] = region_info['coords']
            return cached_result
        
        # 获取对应的提示词
        if region_info["type"] == "table":
            prompt = self.prompt_manager.get_prompt('table_extraction')
        elif region_info["type"] == "figure":
            prompt = self.prompt_manager.get_prompt('figure_analysis')
        else:
            prompt = self.prompt_manager.get_prompt('text_extraction')
        
        # 调用LLM API (支持多模型切换)
        try:
            # 使用LLM管理器获取可用客户端并调用extract_text方法
            result = self.llm_manager.extract_text(region_img, region_info, prompt)
            
            # 保存到缓存
            if 'error' not in result:
                self._save_to_cache(image_hash, cache_type, result)
            
            return result
        except Exception as e:
            error_msg = str(e)
            logger.error(f"处理区域时出错: {error_msg}")
            return {
                'type': region_info["type"],
                'coords': region_info['coords'],
                'text': f"[ERROR] {error_msg}",
                'error': error_msg
            }

    def _process_regions(self, image_np: np.ndarray, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理所有检测到的区域"""
        processed_regions = []
        processed_coords = set()
        
        for region in regions:
            # 确保区域有必要的字段
            if not isinstance(region, dict) or 'coords' not in region or 'type' not in region:
                continue
            
            region_key = f"{region['coords'][0]}_{region['coords'][1]}_{region['coords'][2]}_{region['coords'][3]}"
            
            if region_key in processed_coords:
                continue
            
            # 裁剪区域
            region_img = self._crop_region(image_np, region)
            
            # 处理区域
            processed_region = self._extract_text_from_region(region_img, region)
            
            processed_regions.append(processed_region)
            processed_coords.add(region_key)
            
            # 释放内存
            del region_img
            gc.collect()
        
        return processed_regions

    def _compose_page_raw_text(self, processed_regions: List[Dict[str, Any]]) -> str:
        """按阅读顺序组合页面级原始文本"""
        if not isinstance(processed_regions, list):
            return ""
        
        # 文本类型的区域
        text_like_types = {"plain text", "title", "list", "table", "figure"}
        sortable_regions: List[Tuple[int, int, str]] = []
        
        for region in processed_regions:
            if not isinstance(region, dict):
                continue
            
            region_type = region.get("type")
            if region_type not in text_like_types:
                continue
            
            coords = region.get("coords") or [0, 0, 0, 0]
            try:
                x, y = int(coords[0]), int(coords[1])
            except Exception:
                x, y = 0, 0
            
            text_value = region.get("text", "")
            if isinstance(text_value, str) and text_value.strip():
                # 保留内部换行符，只修剪外部空白
                sortable_regions.append((y, x, text_value.strip()))
        
        # 按y坐标然后x坐标排序
        sortable_regions.sort(key=lambda t: (t[0], t[1]))
        
        # 在区域之间添加空行以分隔块
        return "\n\n".join(t[2] for t in sortable_regions)

    def _determine_pages_to_process(self, total_pages: int, max_pages: Optional[int] = None, 
                                   page_range: Optional[Tuple[int, int]] = None, 
                                   pages: Optional[List[int]] = None) -> List[int]:
        """根据限制选项确定要处理的页面"""
        if pages is not None:
            # 指定了特定页面
            valid_pages = [p for p in pages if 1 <= p <= total_pages]
            if len(valid_pages) != len(pages):
                invalid_pages = [p for p in pages if p not in valid_pages]
                logger.warning(f"无效的页码（不在1-{total_pages}范围内）: {invalid_pages}")
            return sorted(valid_pages)
        elif page_range is not None:
            # 指定了页面范围
            start, end = page_range
            start = max(1, start)
            end = min(total_pages, end)
            return list(range(start, end + 1))
        elif max_pages is not None:
            # 指定了最大页数
            return list(range(1, min(max_pages + 1, total_pages + 1)))
        else:
            # 处理所有页面
            return list(range(1, total_pages + 1))

    def convert_pdf_to_images(self, pdf_path: Union[str, Path], output_folder: Union[str, Path], 
                             dpi: int = 300, max_pages: Optional[int] = None, 
                             page_range: Optional[Tuple[int, int]] = None, 
                             pages: Optional[List[int]] = None) -> List[str]:
        """将PDF文件的指定页转换为图片并保存"""
        pdf_path = Path(pdf_path)
        output_folder = Path(output_folder)
        
        try:
            # 创建输出文件夹
            if not output_folder.exists():
                output_folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"创建文件夹: {output_folder}")
            
            # 获取PDF信息
            try:
                # 首先尝试使用PyMuPDF获取页数信息
                doc = fitz.open(str(pdf_path))
                total_pages = len(doc)
                doc.close()
            except Exception as e:
                logger.warning(f"使用PyMuPDF获取PDF信息失败: {str(e)}，尝试使用pdfinfo_from_path")
                # 回退到使用pdfinfo_from_path
                pdf_info = pdfinfo_from_path(str(pdf_path))
                total_pages = pdf_info['Pages']
            
            # 确定要处理的页面
            pages_to_process = self._determine_pages_to_process(total_pages, max_pages, page_range, pages)
            
            # 将PDF转换为图片列表
            logger.info(f"正在处理PDF文件: {pdf_path}")
            logger.info(f"将处理 {len(pages_to_process)} 页: {pages_to_process}")
            
            start_time = time.time()
            image_paths = []
            
            # 首先尝试使用不需要poppler的PyMuPDF方法
            try:
                image_paths = self._convert_pdf_to_images_with_pymupdf(
                    pdf_path, output_folder, dpi, pages_to_process
                )
                logger.info("使用PyMuPDF成功转换PDF为图片")
                
                processing_time = time.time() - start_time
                logger.info(f"转换PDF为图片完成，耗时: {processing_time:.2f}秒，成功转换页数: {len(image_paths)}")
                
                return image_paths
            except Exception as e:
                logger.warning(f"使用PyMuPDF转换PDF为图片失败: {str(e)}，尝试使用poppler方案")
            
            # 如果PyMuPDF方案失败，回退到使用poppler的方案
            logger.info("回退到使用poppler方案转换PDF为图片")
            
            # 设置poppler路径，支持中文显示
            # 在Windows系统上尝试自动检测poppler路径
            poppler_path = None
            if os.name == 'nt':  # Windows系统
                # 尝试常见的poppler安装路径
                common_poppler_paths = [
                    r'C:\Program Files\poppler-0.68.0\bin',
                    r'C:\Program Files (x86)\poppler-0.68.0\bin',
                    r'C:\poppler\bin',
                    r'C:\Program Files\poppler\bin',
                    r'C:\Users\$USERNAME\AppData\Local\poppler\bin'
                ]
                
                # 检查环境变量中的POPPLER_PATH
                env_poppler_path = os.environ.get('POPPLER_PATH')
                if env_poppler_path:
                    common_poppler_paths.insert(0, env_poppler_path)
                
                for path in common_poppler_paths:
                    # 替换$USERNAME为当前用户名
                    if '$USERNAME' in path:
                        path = path.replace('$USERNAME', os.environ.get('USERNAME', ''))
                    
                    if os.path.exists(path):
                        poppler_path = path
                        logger.info(f"找到poppler路径: {poppler_path}")
                        break
            
            # 为PDF转换设置参数，支持中文显示
            pdf_to_image_kwargs = {
                'first_page': None, 
                'last_page': None, 
                'dpi': dpi,
                'poppler_path': poppler_path,
                # 设置参数以改善中文显示
                'thread_count': 1,
                'use_cropbox': True,
                'fmt': 'png',
                'grayscale': False,
                'size': None,
                'transparent': False
            }
            
            for page_num in pages_to_process:
                try:
                    # 只转换指定页
                    images = convert_from_path(
                        str(pdf_path), 
                        first_page=page_num, 
                        last_page=page_num, 
                        **pdf_to_image_kwargs
                    )
                    image = images[0]
                    
                    # 保存图片
                    image_path = output_folder / f"page_{page_num}.png"
                    image.save(str(image_path), "PNG")
                    image_paths.append(str(image_path))
                    logger.debug(f"已保存图片: {image_path}")
                    
                    # 释放内存
                    del images, image
                    gc.collect()
                except Exception as e:
                    logger.error(f"转换第 {page_num} 页时出错: {str(e)}")
                    # 尝试在出错时提供更详细的错误信息
                    if 'poppler' in str(e).lower():
                        logger.error("请确保已安装poppler并正确设置了路径。可以从https://github.com/oschwartz10612/poppler-windows/releases/下载")
                    elif 'font' in str(e).lower() or '字体' in str(e):
                        logger.error("中文显示问题可能是由于缺少中文字体支持。请确保系统中安装了常用中文字体。")
            
            processing_time = time.time() - start_time
            logger.info(f"转换PDF为图片完成，耗时: {processing_time:.2f}秒，成功转换页数: {len(image_paths)}")
            
            return image_paths
        except Exception as e:
            logger.error(f"处理PDF文件时出错: {str(e)}")
            # 如果错误与poppler相关，提供更详细的指导
            if 'poppler' in str(e).lower():
                logger.error("请确保已安装poppler并正确设置了路径。可以从https://github.com/oschwartz10612/poppler-windows/releases/下载")
            elif 'font' in str(e).lower() or '字体' in str(e):
                logger.error("中文显示问题可能是由于缺少中文字体支持。请确保系统中安装了常用中文字体。")
            raise
            
    def _convert_pdf_to_images_with_pymupdf(self, pdf_path: Path, output_folder: Path, 
                                          dpi: int, pages_to_process: List[int]) -> List[str]:
        """使用PyMuPDF（fitz）将PDF转换为图片，不需要poppler"""
        image_paths = []
        
        # 打开PDF文件
        doc = fitz.open(str(pdf_path))
        
        try:
            # 遍历要处理的页面
            for page_num in pages_to_process:
                # PyMuPDF的页码从0开始
                fitz_page_num = page_num - 1
                
                # 检查页码是否有效
                if fitz_page_num < 0 or fitz_page_num >= len(doc):
                    logger.warning(f"页码 {page_num} 无效，跳过")
                    continue
                
                # 获取页面
                page = doc[fitz_page_num]
                
                # 设置转换参数，提高DPI以获得更好的图像质量
                zoom = dpi / 72  # PDF默认DPI为72
                mat = fitz.Matrix(zoom, zoom)
                
                # 将页面转换为图像
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                
                # 保存图像为PNG
                image_path = output_folder / f"page_{page_num}.png"
                pix.save(str(image_path))
                
                image_paths.append(str(image_path))
                logger.debug(f"已使用PyMuPDF保存图片: {image_path}")
                
                # 释放内存
                del page, pix
                gc.collect()
        finally:
            # 关闭文档
            doc.close()
        
        return image_paths

    def process_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """处理单张图片，进行布局检测和内容提取"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        logger.info(f"正在处理图片: {image_path}")
        
        # 加载图像
        image_np = cv2.imread(str(image_path))
        if image_np is None:
            raise ValueError(f"无法加载图片: {image_path}")
        
        # 直接处理整页图片而不分块
        if self.use_full_page:
            logger.info(f"使用整页处理模式")
            
            # 创建一个覆盖整个页面的区域
            height, width = image_np.shape[:2]
            full_page_region = {"type": "plain text", "coords": [0, 0, width, height], "confidence": 1.0}
            
            # 直接处理整页图片
            processed_region = self._extract_text_from_region(image_np, full_page_region)
            
            result = {
                'image_path': str(image_path),
                'regions': [processed_region],
                'raw_text': processed_region.get('text', ''),
                'processed_at': datetime.now().isoformat()
            }
            
            return result
        
        # 常规分块处理
        # 检测布局
        regions = []
        if self.layout_model:
            try:
                regions = self.layout_model.predict(image_np, conf=self.confidence_threshold)
                logger.info(f"检测到 {len(regions)} 个区域")
            except Exception as e:
                logger.warning(f"布局检测失败，将使用整页处理: {str(e)}")
        
        # 如果没有检测到区域或布局模型不可用，创建一个覆盖整个页面的区域
        if not regions:
            height, width = image_np.shape[:2]
            regions = [{"type": "plain text", "coords": [0, 0, width, height], "confidence": 1.0}]
        
        # 处理区域
        processed_regions = self._process_regions(image_np, regions)
        
        # 组合页面文本
        raw_text = self._compose_page_raw_text(processed_regions)
        
        result = {
            'image_path': str(image_path),
            'regions': processed_regions,
            'raw_text': raw_text,
            'processed_at': datetime.now().isoformat()
        }
        
        return result

    def process_pdf_page(self, pdf_path: Union[str, Path], page_num: int, dpi: int = 300, use_full_page: bool = None) -> Dict[str, Any]:
        """处理PDF的单页"""
        pdf_path = Path(pdf_path)
        
        try:
            logger.info(f"正在处理PDF: {pdf_path} 的第 {page_num} 页")
            
            # 使用PyMuPDF直接读取PDF页面，完全替代convert_from_path
            doc = fitz.open(str(pdf_path))
            try:
                # PyMuPDF的页码从0开始
                fitz_page_num = page_num - 1
                
                # 检查页码是否有效
                if fitz_page_num < 0 or fitz_page_num >= len(doc):
                    logger.warning(f"页码 {page_num} 无效")
                    return {
                        'page_number': page_num,
                        'error': f"页码 {page_num} 无效",
                        'processed_at': datetime.now().isoformat()
                    }
                
                # 获取页面
                page = doc[fitz_page_num]
                
                # 设置转换参数，提高DPI以获得更好的图像质量
                zoom = dpi / 72  # PDF默认DPI为72
                mat = fitz.Matrix(zoom, zoom)
                
                # 将页面转换为图像
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                
                # 保存临时图像
                temp_image_path = self.temp_dir / f"{pdf_path.stem}_page_{page_num}.png"
                pix.save(str(temp_image_path))
                
                # 释放页面资源
                del page, pix
            finally:
                # 关闭文档
                doc.close()
            
            # 处理图像
            # 如果传入了use_full_page参数，则临时覆盖实例的use_full_page设置
            if use_full_page is not None:
                original_use_full_page = self.use_full_page
                self.use_full_page = use_full_page
                result = self.process_image(temp_image_path)
                self.use_full_page = original_use_full_page
            else:
                result = self.process_image(temp_image_path)
            
            result['page_number'] = page_num
            
            # 释放内存
            gc.collect()
            
            return result
        except Exception as e:
            logger.error(f"处理第 {page_num} 页时出错: {str(e)}")
            return {
                'page_number': page_num,
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }

    def process_pdf(self, pdf_path: Union[str, Path], output_folder: Union[str, Path], 
                   dpi: int = 300, max_pages: Optional[int] = None, 
                   page_range: Optional[Tuple[int, int]] = None, 
                   pages: Optional[List[int]] = None, 
                   use_full_page: bool = None) -> str:
        """处理PDF文件并生成Markdown"""
        pdf_path = Path(pdf_path)
        output_folder = Path(output_folder)
        pdf_name = pdf_path.stem
        
        try:
            # 获取PDF信息
            pdf_info = pdfinfo_from_path(str(pdf_path))
            total_pages = pdf_info['Pages']
            
            # 确定要处理的页面
            pages_to_process = self._determine_pages_to_process(total_pages, max_pages, page_range, pages)
            
            logger.info(f"开始处理文件: {pdf_path}")
            logger.info(f"总页数: {total_pages}")
            logger.info(f"将处理 {len(pages_to_process)} 页")
            
            # 创建Markdown内容
            markdown_parts = []
            markdown_parts.append(f"# PDF文档内容提取\n")
            markdown_parts.append(f"> 提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            markdown_parts.append(f"> 总页数: {total_pages}\n")
            markdown_parts.append(f"> 处理页数: {len(pages_to_process)}\n\n")
            
            # 处理每一页
            for page_num in pages_to_process:
                logger.info(f"正在处理第 {page_num}/{total_pages} 页")
                
                # 添加分页标题
                markdown_parts.append(f"## 第 {page_num} 页\n\n")
                
                try:
                    # 处理页面
                    start_time = time.time()
                    page_result = self.process_pdf_page(pdf_path, page_num, dpi, use_full_page)
                    processing_time = time.time() - start_time
                    
                    logger.info(f"处理第 {page_num} 页耗时: {processing_time:.2f}秒")
                    
                    # 添加处理结果
                    if 'error' in page_result:
                        markdown_parts.append(f"[错误] 处理此页时出错: {page_result['error']}\n\n")
                    else:
                        markdown_parts.append(f"{page_result['raw_text']}\n\n")
                    
                except Exception as e:
                    logger.error(f"处理第 {page_num} 页时出错: {str(e)}")
                    markdown_parts.append(f"[错误] 处理此页时出错: {str(e)}\n\n")
                
                # 添加分隔线
                markdown_parts.append("---\n\n")
            
            # 组合最终Markdown内容
            markdown_content = ''.join(markdown_parts)
            
            # 保存Markdown文件
            md_file_path = output_folder / f"{pdf_name}.md"
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"已保存Markdown文件: {md_file_path}")
            
            return str(md_file_path)
        except Exception as e:
            logger.error(f"处理文件 {pdf_path} 时出错: {str(e)}")
            raise

    def process_directory(self, directory: Union[str, Path], dpi: int = 300, max_pages: Optional[int] = None, 
                         page_range: Optional[Tuple[int, int]] = None, 
                         pages: Optional[List[int]] = None, 
                         use_full_page: bool = None) -> Dict[str, Any]:
        """递归处理目录中的所有PDF文件"""
        directory = Path(directory)
        
        # 创建统一的output文件夹
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        if not directory.exists():
            logger.error(f"目录不存在: {directory}")
            return {"error": f"目录不存在: {directory}"}
        
        total_files = 0
        success_count = 0
        skipped_count = 0
        failed_files = []
        
        # 递归遍历目录
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    total_files += 1
                    pdf_path = Path(root) / file
                    pdf_name = os.path.splitext(file)[0]
                    
                    try:
                        logger.info(f"开始处理文件 {total_files}: {pdf_path}")
                        
                        # 检查Markdown文件是否已存在于output文件夹中
                        md_file_path = output_dir / f"{pdf_name}.md"
                        if md_file_path.exists():
                            logger.info(f"Markdown文件已存在，跳过处理: {md_file_path}")
                            skipped_count += 1
                            continue
                        
                        # 处理PDF并生成Markdown
                        self.process_pdf(pdf_path, output_dir, dpi, max_pages, page_range, pages, use_full_page)
                        
                        success_count += 1
                        
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"处理文件 {pdf_path} 时出错: {error_msg}")
                        failed_files.append({"path": str(pdf_path), "error": error_msg})
        
        logger.info(f"\n处理完成！")
        logger.info(f"总PDF文件数: {total_files}")
        logger.info(f"成功转换: {success_count}")
        logger.info(f"已存在跳过: {skipped_count}")
        logger.info(f"转换失败: {len(failed_files)}")
        
        return {
            "total_files": total_files,
            "success_count": success_count,
            "skipped_count": skipped_count,
            "failed_count": len(failed_files),
            "failed_files": failed_files
        }

def process_images_with_gemini(image_paths, gemini_processor):
    """使用Gemini处理图片列表并生成Markdown内容"""
    markdown_content = """
# PDF文档内容提取
> 提取时间: {}
> 总页数: {}

""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), len(image_paths))
    
    # 处理每一页图片
    for i, image_path in enumerate(image_paths):
        page_num = i + 1
        logger.info(f"正在处理第 {page_num}/{len(image_paths)} 页")
        
        try:
            # 添加分页标题
            markdown_content += f"## 第 {page_num} 页\n\n"
            
            # 使用Gemini处理图片
            start_time = time.time()
            content = gemini_processor.process_image(image_path)
            processing_time = time.time() - start_time
            
            logger.info(f"处理第 {page_num} 页耗时: {processing_time:.2f}秒")
            
            # 提取原始文本
            if isinstance(content, dict) and 'raw_text' in content:
                markdown_content += f"{content['raw_text']}\n\n"
            elif isinstance(content, str):
                markdown_content += f"{content}\n\n"
            else:
                markdown_content += "[处理结果为空]\n\n"
            
            # 添加分隔线
            markdown_content += "---\n\n"
            
        except Exception as e:
            logger.error(f"处理第 {page_num} 页时出错: {str(e)}")
            markdown_content += f"[错误] 处理此页时出错: {str(e)}\n\n---\n\n"
    
    return markdown_content


def parse_page_range(page_range_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """解析页面范围参数"""
    if not page_range_str:
        return None
    
    try:
        if '-' in page_range_str:
            start, end = map(int, page_range_str.split('-', 1))
            if start < 1 or end < start:
                logger.warning(f"无效的页面范围: {page_range_str}")
                return None
            return (start, end)
        else:
            page = int(page_range_str)
            if page < 1:
                logger.warning(f"无效的页码: {page}")
                return None
            return (page, page)
    except ValueError:
        logger.warning(f"无法解析页面范围: {page_range_str}")
        return None


def parse_specific_pages(pages_str: Optional[str]) -> Optional[List[int]]:
    """解析特定页面参数"""
    if not pages_str:
        return None
    
    try:
        # 支持逗号分隔的页码列表
        if ',' in pages_str:
            pages = [int(p.strip()) for p in pages_str.split(',') if p.strip()]
        # 支持空格分隔的页码列表
        elif ' ' in pages_str:
            pages = [int(p) for p in pages_str.split() if p.isdigit()]
        else:
            pages = [int(pages_str)]
        
        # 验证页码
        valid_pages = [p for p in pages if p >= 1]
        if len(valid_pages) != len(pages):
            invalid_pages = [p for p in pages if p < 1]
            logger.warning(f"无效的页码（小于1）: {invalid_pages}")
        
        return valid_pages
    except ValueError:
        logger.warning(f"无法解析特定页面: {pages_str}")
        return None


def main():
    """主函数，处理命令行参数并执行PDF转换"""
    parser = argparse.ArgumentParser(description='将PDF文件转换为Markdown文本')
    parser.add_argument('--dir', type=str, help='要处理的目录路径')
    parser.add_argument('--file', type=str, help='要处理的单个PDF文件路径')
    parser.add_argument('--api-key', type=str, help='OpenRouter API密钥')
    parser.add_argument('--model', type=str, default='google/gemini-2.5-flash', help='使用的OpenRouter模型，默认google/gemini-2.5-flash，需要使用完整格式如"google/gemini-2.5-flash"')
    parser.add_argument('--dpi', type=int, default=300, help='PDF转换为图片的DPI值，默认300')
    parser.add_argument('--max-pages', type=int, help='处理的最大页数')
    parser.add_argument('--page-range', type=str, help='处理的页面范围，如1-10')
    parser.add_argument('--pages', type=str, help='处理的特定页面，如1,3,5或1 3 5')
    parser.add_argument('--no-cache', action='store_true', help='禁用结果缓存')
    parser.add_argument('--confidence', type=float, default=0.5, help='布局检测的置信度阈值，默认0.5')
    parser.add_argument('--full-page', action='store_true', help='不分块，直接将整页图片送给模型处理')
    
    args = parser.parse_args()
    
    # 解析页面参数
    max_pages = args.max_pages
    page_range = parse_page_range(args.page_range)
    pages = parse_specific_pages(args.pages)
    
    # 初始化PDF处理器
    processor = PDFProcessor(
        api_key=args.api_key,
        model=args.model,
        confidence_threshold=args.confidence,
        use_cache=not args.no_cache,
        use_full_page=args.full_page
    )
    
    try:
        # 处理单个文件
        if args.file:
            file_path = Path(args.file)
            if not file_path.exists():
                logger.error(f"文件不存在: {file_path}")
                return
            
            if not file_path.suffix.lower() == '.pdf':
                logger.error(f"不是PDF文件: {file_path}")
                return
            
            # 创建与PDF文件同名的文件夹作为输出目录
            output_folder = file_path.parent / file_path.stem
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # 处理PDF文件
            md_file_path = processor.process_pdf(
                file_path,
                output_folder,
                dpi=args.dpi,
                max_pages=max_pages,
                page_range=page_range,
                pages=pages,
                use_full_page=args.full_page
            )
            
            logger.info(f"PDF文件处理完成，生成的Markdown文件: {md_file_path}")
            
        # 处理目录
        elif args.dir:
            # 处理目录
            result = processor.process_directory(
                args.dir,
                dpi=args.dpi,
                max_pages=max_pages,
                page_range=page_range,
                pages=pages,
                use_full_page=args.full_page
            )
            
            # 输出处理结果
            if 'error' not in result:
                logger.info(f"\n目录处理完成！")
                logger.info(f"总PDF文件数: {result['total_files']}")
                logger.info(f"成功转换: {result['success_count']}")
                logger.info(f"转换失败: {result['failed_count']}")
            else:
                logger.error(result['error'])
        
        else:
            logger.error("请提供要处理的目录路径(--dir)或单个文件路径(--file)")
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("用户中断了处理")
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")


if __name__ == "__main__":
    main()