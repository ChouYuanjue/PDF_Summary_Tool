# PDF Summary Tool

一个强大的Python工具，用于将PDF文档转换为结构化文本，并利用先进的AI模型生成高质量的内容总结和关键词提炼。

本项目大量参考了项目[NoUnique/vlm-ocr-pipeline](https://github.com/NoUnique/vlm-ocr-pipeline)的实现逻辑，同时在项目[ChouYuanjue/Informal_Notes_on_Mathematics](https://github.com/ChouYuanjue/Informal_Notes_on_Mathematics)上进行了实验，可移步该项目查看本项目的处理效果。

## 功能特点

- **PDF转图片处理**：递归扫描指定目录及其子目录中的所有PDF文件，转换为图片格式
- **高级OCR识别**：使用Google Gemini Vision Language Model进行精确文本识别
- **多模型支持**：集成Google Gemini、OpenAI以及SiliconCloud等多种AI服务
- **Markdown生成**：将识别结果保存为格式良好的Markdown文件
- **智能总结**：自动生成文档摘要和关键词提取
- **自定义配置**：支持DPI设置、输出路径自定义等参数调整
- **错误处理与日志**：完善的错误处理机制和详细的日志记录

## 项目结构

```
PDF_Summary_Tool/
├── .cache/           # 缓存文件
├── .tmp/             # 临时图片文件
├── models/           # 模型文件
├── output/           # Markdown输出
├── pipeline/         # LLM客户端和处理管道
│   ├── gemini.py     # Google Gemini API集成
│   ├── llm_client.py # LLM客户端基类
│   ├── openai.py     # OpenAI API集成
│   └── prompt.py     # 提示词模板
├── yaml_output/      # 总结文档的YAML格式输出
├── pdf_to_md.py      # PDF转Markdown主脚本
├── md_to_summary.py  # Markdown转总结主脚本
├── requirements.txt  # 项目依赖
└── yolov8n.pt        # YOLO模型文件
```

## 安装说明

### 1. 环境要求
- Python 3.8或更高版本
- 足够的磁盘空间用于存储PDF、临时图片和输出文件

### 2. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 3. 安装Poppler（PDF转图片所需）

- **Windows**：从[Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)下载，解压后将bin目录添加到系统PATH环境变量中
- **macOS**：使用Homebrew安装 `brew install poppler`
- **Linux**：使用包管理器安装，如Ubuntu: `sudo apt-get install poppler-utils`

### 4. 获取API密钥

项目需要以下API密钥之一或多个：
- **Google Gemini API**：访问[Google AI Studio](https://makersuite.google.com/app/apikey)获取
- **SiliconFlow API**：访问[SiliconFlow官网](https://www.siliconflow.cn/)获取
- **OpenAI API**（可选）：访问[OpenAI官网](https://platform.openai.com/)获取

## 环境配置

推荐通过`.env`文件设置API密钥：

1. 创建`.env`文件
2. 在文件中填入您的API密钥：
   
   ```env
   # Gemini API密钥（用于PDF转图片后的OCR处理）
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # SiliconFlow API密钥（用于Markdown文件生成英文总结）
   SILICONFLOW_API_KEY=your_siliconflow_api_key_here
   
   # OpenAI API密钥（可选，用于替代Gemini进行OCR或总结）
   OPENAI_API_KEY=your_openai_api_key_here
   
   # OpenRouter API密钥（可选，用于访问其他模型）
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

## 使用方法

### PDF转Markdown

将PDF文件转换为Markdown格式，提取其中的文本内容：

```bash
python pdf_to_md.py --dir <directory_path> [options]
# 或处理单个文件
python pdf_to_md.py --file <pdf_file_path> [options]
```

主要选项：
- `--dir`: 指定包含PDF文件的目录
- `--file`: 指定要处理的单个PDF文件路径
- `--api-key`: 提供OpenRouter API密钥
- `--model`: 使用的OpenRouter模型（默认：google/gemini-2.5-flash）
- `--dpi`: 设置转换图片的DPI值（默认：300）
- `--max-pages`: 处理的最大页数
- `--page-range`: 处理的页面范围（如1-10）
- `--pages`: 处理的特定页面（如1,3,5或1 3 5）
- `--no-cache`: 禁用结果缓存
- `--confidence`: 布局检测的置信度阈值（默认：0.5）
- `--full-page`: 不分块，直接将整页图片送给模型处理

### Markdown转总结

将生成的Markdown文件转换为总结和关键词：

```bash
python md_to_summary.py <directory_path> [--api-key <api_key>] [--model <model_name>]
```

主要选项：
- `<directory_path>`: 要处理的Markdown文件所在目录（必需）
- `--api-key`: 提供SiliconFlow API密钥（如果未设置环境变量）
- `--model`: 使用的LLM模型（默认：deepseek-ai/DeepSeek-V3）

## 高级配置

### 模型选择

在`md_to_summary.py`中，系统会按以下优先级使用AI模型：
- **OpenRouter**：默认使用`deepseek/deepseek-chat-v3.1:free`模型（如果设置了OPENROUTER_API_KEY）
- **SiliconFlow**：如果未配置OpenRouter，则使用`deepseek-ai/DeepSeek-V3`或指定的其他模型
- **Gemini**：如果未配置OpenRouter和SiliconFlow，可使用`gemini-2.5-flash`模型
- **OpenAI**：在`pdf_to_md.py`中可选择使用OpenAI的模型（如`gpt-5-nano`）

## 一键完成完整流程

使用`pdf_to_summary_pipeline.py`可以一键完成从PDF到Markdown再到英文总结的完整流程：

```bash
python pdf_to_summary_pipeline.py <directory_path> [--gemini-api-key <gemini_api_key>] [--siliconflow-api-key <siliconflow_api_key>] [--dpi <dpi_value>] [--model <model_name>] [--full-page]
```

主要选项：
- `<directory_path>`: 要处理的PDF文件所在目录（必需）
- `--gemini-api-key`: Gemini API密钥（用于PDF转Markdown阶段）
- `--siliconflow-api-key`: SiliconFlow API密钥（用于Markdown转总结阶段）
- `--dpi`: 输出图片的DPI值（默认：300）
- `--model`: 使用的LLM模型（默认：deepseek-ai/DeepSeek-V3）
- `--full-page`: 不分块，直接将整页图片送给模型处理

**示例用法**：
```bash
python pdf_to_summary_pipeline.py ./pdf_files --gemini-api-key YOUR_GEMINI_KEY --siliconflow-api-key YOUR_SILICONFLOW_KEY --dpi 300
```

这个脚本会自动完成以下两个阶段：
1. **PDF转Markdown**：使用Gemini模型将PDF文件转换为Markdown格式
2. **Markdown转总结**：使用指定的LLM模型从Markdown生成英文总结和提取关键词

生成的.md和.yaml文件将分别保存到`output`和`yaml_output`文件夹中。

### 处理参数调整

- **图片质量**：通过`--dpi`参数调整，更高的DPI值会生成更清晰的图片，但处理时间更长
- **批处理大小**：可在代码中调整批处理参数以优化性能

## 常见问题解决

### 缺少API密钥错误
确保`.env`文件中的API密钥正确设置，或者通过命令行参数提供。

### Poppler未找到错误
确认Poppler已正确安装并添加到系统PATH环境变量中。

### 内存不足错误
对于大型PDF文件，可能需要增加系统内存或调整代码中的批处理参数。

## 依赖说明

项目主要依赖库包括：
- **PyMuPDF**：PDF处理库
- **pdf2image**：PDF转图片工具
- **Pillow**：图像处理库
- **google-generativeai**：Google Gemini API客户端
- **openai**：OpenAI API客户端
- **python-dotenv**：环境变量加载工具
- **ultralytics**：YOLO模型支持
- **torch**：PyTorch深度学习框架

## 贡献指南

欢迎提交问题和改进建议。如有重大修改，请先创建issue讨论。

## 许可证

[MIT License](LICENSE)
