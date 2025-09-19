import os
import logging
import time
import argparse
import os
from datetime import datetime
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdf_to_summary_pipeline.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 导入现有脚本的功能
try:
    # 导入pdf_to_md.py的功能
    import pdf_to_md
    from pdf_to_md import PDFProcessor
    # 导入md_to_summary.py的功能
    import md_to_summary
except ImportError as e:
    logger.error(f"导入模块失败: {str(e)}")
    raise

def process_pdf_to_summary_pipeline(directory, gemini_api_key=None, siliconflow_api_key=None, dpi=300, model='deepseek-ai/DeepSeek-V3', use_full_page=False):
    """一键处理PDF文件转换为Markdown并生成英文总结和关键词"""
    if not os.path.exists(directory):
        logger.error(f"目录不存在: {directory}")
        return
    
    # 确保output目录存在
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"已创建output目录: {output_dir}")
    
    start_time = time.time()
    logger.info(f"=== 开始PDF到英文总结的完整流程 ===")
    logger.info(f"处理目录: {directory}")
    logger.info(f"设置的DPI: {dpi}")
    logger.info(f"使用的模型: {model}")
    logger.info(f"输出目录: {output_dir}")
    
    # 第一阶段: PDF转Markdown
    logger.info("\n=== 第一阶段: PDF转换为Markdown ===")
    try:
        # 初始化PDFProcessor并调用其process_directory方法
        processor = PDFProcessor(api_key=gemini_api_key, use_full_page=use_full_page)
        processor.process_directory(directory, dpi=dpi, use_full_page=use_full_page)
        logger.info("PDF转换为Markdown完成，所有Markdown文件已保存到output文件夹")
    except Exception as e:
        logger.error(f"PDF转换为Markdown时出错: {str(e)}")
        return
    
    # 第二阶段: Markdown生成英文总结和提取关键词
    logger.info("\n=== 第二阶段: Markdown生成英文总结和提取关键词 ===")
    try:
        # 调用md_to_summary.py的process_directory函数
        md_to_summary.process_directory(directory, api_key=siliconflow_api_key, model=model)
        logger.info("Markdown生成英文总结和提取关键词完成，所有YAML文件已保存到output文件夹")
    except Exception as e:
        logger.error(f"Markdown生成英文总结和提取关键词时出错: {str(e)}")
        return
    
    total_time = time.time() - start_time
    logger.info(f"\n=== 完整流程处理完成 ===")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info(f"处理目录: {directory}")
    logger.info(f"所有生成的.md和.yaml文件已保存到: {output_dir}")


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='一键完成PDF转换为Markdown并生成英文总结和关键词的完整流程')
    parser.add_argument('directory', help='要处理的目录路径')
    parser.add_argument('--gemini-api-key', help='Gemini API密钥（如果未设置环境变量）')
    parser.add_argument('--siliconflow-api-key', help='SiliconFlow API密钥（如果未设置环境变量）')
    parser.add_argument('--dpi', type=int, default=300, help='输出图片的DPI（默认: 300）')
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-V3', help='使用的LLM模型（默认: deepseek-ai/DeepSeek-V3）')
    parser.add_argument('--full-page', action='store_true', help='不分块，直接将整页图片送给模型处理')
    
    args = parser.parse_args()
    
    # 执行完整流程
    process_pdf_to_summary_pipeline(
        args.directory,
        gemini_api_key=args.gemini_api_key,
        siliconflow_api_key=args.siliconflow_api_key,
        dpi=args.dpi,
        model=args.model,
        use_full_page=args.full_page
    )


if __name__ == "__main__":
    main()