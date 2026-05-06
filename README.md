# 批量图片 AI 人像处理（分步骤运行）

这个项目提供一个可一步一步执行的 Python 程序：

1. 读取文件夹全部图片
2. 自动 AI 分析人物（人脸检测 + 数量统计）
3. 自动美颜（磨皮 + 提亮）
4. 自动调色（白平衡 + 对比度 + 饱和度）
5. 输出到新文件夹

## 1) 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 准备输入目录

例如：

```bash
mkdir -p input_images output_images
# 把你的图片放到 input_images
```

## 3) 运行程序

```bash
python batch_portrait_pipeline.py --input input_images --output output_images --report analysis_report.json
```

## 4) 查看结果

- 处理后的图片在 `output_images/`
- AI 分析结果在 `analysis_report.json`

## 参数说明

- `--input`: 输入图片目录
- `--output`: 输出图片目录（不存在会自动创建）
- `--report`: 分析报告 JSON 路径（默认 `analysis_report.json`）
