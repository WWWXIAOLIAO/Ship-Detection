# 第三方来源

| 组件 | 用途 | 来源 / 许可证 |
|---|---|---|
| Ultralytics YOLO11 | 训练、预测、NMS | [ultralytics](https://github.com/ultralytics/ultralytics)，AGPL-3.0 |
| OpenAI CLIP ViT-B/32 | 冻结的图像与文本编码器 | [CLIP](https://github.com/openai/CLIP)，MIT |
| OpenCLIP | 加载及运行 CLIP | [open_clip](https://github.com/mlfoundations/open_clip)，MIT |
| COCO API / pycocotools | 检测指标 | [cocoapi](https://github.com/cocodataset/cocoapi)，BSD |

`ScorePredictor` 的后处理流程参考 Ultralytics `DetectionPredictor`，额外保留 NMS 选中候选的完整类别得分，用于文本后融合。

教师数据、原始文档、图片、知识库与微调权重不随代码仓库发布。原始 CLIP 权重由工具从公开地址下载，并校验 SHA256：

```text
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
```

`docs/figures/` 为实验统计图和方法图；不含教师数据库中的原始照片。`results/reference/` 为已完成实验的数字记录。
