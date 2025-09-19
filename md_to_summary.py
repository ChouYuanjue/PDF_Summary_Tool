import argparse
import os
import sys
import yaml
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# 导入LLM客户端模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.llm_client import GeminiLLMClient, SiliconCloudClient, LLMClientManager
from pipeline.openai import OpenAIClient

# 加载.env文件中的环境变量
load_dotenv()

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('md_to_summary.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def setup_llm_clients(api_key=None, model=None):
    """设置LLM客户端管理器"""
    llm_manager = LLMClientManager()
    
    # 首先注册OpenRouter客户端使用deepseek模型作为主要客户端
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_api_key:
        openrouter_client = OpenAIClient(
            model='deepseek/deepseek-chat-v3.1:free',
            api_key=openrouter_api_key,
            base_url='https://openrouter.ai/api/v1'
        )
        llm_manager.register_client(openrouter_client)
    else:
        logger.warning("OPENROUTER_API_KEY环境变量未设置，无法使用OpenRouter客户端")
    
    # 然后注册SiliconCloud客户端作为备用客户端
    siliconcloud_api_key = api_key or os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("SILICONCLOUD_API_KEY")
    if siliconcloud_api_key:
        siliconcloud_client = SiliconCloudClient(
            api_key=siliconcloud_api_key,
            model=model or 'deepseek-ai/DeepSeek-V3'
        )
        llm_manager.register_client(siliconcloud_client)
    else:
        logger.warning("SILICONFLOW_API_KEY/SILICONCLOUD_API_KEY环境变量未设置，无法使用SiliconCloud客户端")
    
    return llm_manager


def load_markdown_file(file_path):
    """加载Markdown文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"已加载文件: {file_path}")
        return content
    except Exception as e:
        logger.error(f"加载文件时出错 {file_path}: {str(e)}")
        return None


def save_yaml_file(file_path, data):
    """保存YAML文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)
        logger.info(f"已保存YAML文件: {file_path}")
    except Exception as e:
        logger.error(f"保存文件时出错 {file_path}: {str(e)}")


def process_directory(directory, api_key=None, model=None):
    """递归处理目录中的所有Markdown文件"""
    # 使用output文件夹作为处理目录
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        logger.error(f"output目录不存在: {output_dir}")
        return
    
    # 创建新的文件夹来存储YAML文件
    yaml_dir = os.path.join(os.getcwd(), 'yaml_output')
    if not os.path.exists(yaml_dir):
        os.makedirs(yaml_dir)
        logger.info(f"创建YAML输出目录: {yaml_dir}")
        
    total_files = 0
    success_count = 0
    failed_count = 0
    
    # 设置LLM客户端管理器
    llm_manager = setup_llm_clients(api_key=api_key, model=model)
    
    if not llm_manager.get_available_clients():
        logger.error("没有可用的LLM API客户端，请确保已设置至少一个API密钥")
        return
        
    # 只处理output文件夹中的Markdown文件
    for file in os.listdir(output_dir):
        if file.lower().endswith('.md') and not file.startswith('.'):
            total_files += 1
            md_path = os.path.join(output_dir, file)
            yaml_path = os.path.join(yaml_dir, f"{os.path.splitext(file)[0]}.yaml")
            
            # 检查YAML文件是否已存在
            if os.path.exists(yaml_path):
                logger.info(f"YAML文件已存在，跳过处理: {yaml_path}")
                continue
                     
            try:
                logger.info(f"开始处理文件 {total_files}: {md_path}")
                
                # 加载Markdown内容
                md_content = load_markdown_file(md_path)
                if not md_content:
                    failed_count += 1
                    continue
                     
                # 为了避免API调用过于频繁，这里限制内容长度
                max_content_length = 15000
                if len(md_content) > max_content_length:
                    logger.info(f"文档内容过长({len(md_content)}字符)，截断为{max_content_length}字符")
                    md_content = md_content[:max_content_length] + "\n[内容过长，已截断]"
                     
                # 生成英文总结
                summary_prompt = "You are an expert in academic research. Please summarize the following document in English. The summary should be comprehensive, capturing all key points and main ideas. Focus on the research content, findings, and conclusions."
                summary = llm_manager.generate_summary(md_content, summary_prompt)
                
                # 提取关键词
                keywords_prompt = "Based on the following document, extract the main keywords, especially focusing on research branches, technologies, methodologies, and important concepts mentioned. Please provide them as a list of keywords separated by commas."
                keywords_text = llm_manager.generate_summary(md_content, keywords_prompt)
                
                # 处理关键词结果
                keywords = []
                if keywords_text:
                    # 清理关键词列表
                    keywords = [kw.strip() for kw in keywords_text.split(',')]
                    # 过滤空字符串
                    keywords = [kw for kw in keywords if kw]
                     
                # 获取使用的模型信息
                used_client = llm_manager.last_used_client
                model_used = used_client.model if used_client else "未知模型"
                
                # 构建YAML数据
                yaml_data = {
                    'document': file,
                    'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_used': model_used,
                    'summary': summary,
                    'keywords': keywords
                }
                
                # 保存YAML文件
                save_yaml_file(yaml_path, yaml_data)
                
                success_count += 1
                
                # 避免API调用过于频繁
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"处理文件 {md_path} 时出错: {str(e)}")
                failed_count += 1
                     
    logger.info(f"\n处理完成！")
    logger.info(f"总Markdown文件数: {total_files}")
    logger.info(f"成功处理: {success_count}")
    logger.info(f"处理失败: {failed_count}")


if __name__ == "__main__":
    # 确保导入argparse
    import argparse
    
    def main():
        """主函数，处理命令行参数"""
        parser = argparse.ArgumentParser(description='将指定目录下的所有Markdown文件通过SiliconFlow LLM生成英文总结并提取关键词，输出YAML文件')
        parser.add_argument('directory', help='要处理的目录路径')
        parser.add_argument('--api-key', help='SiliconFlow API密钥（如果未设置环境变量）')
        parser.add_argument('--model', default='', help='使用的LLM模型（默认: deepseek-ai/DeepSeek-V3）')
        
        args = parser.parse_args()
        
        logger.info(f"开始处理目录: {args.directory}")
        logger.info(f"使用的模型: {args.model}")
        
        # 处理目录
        process_directory(args.directory, api_key=args.api_key, model=args.model)
    
    main()