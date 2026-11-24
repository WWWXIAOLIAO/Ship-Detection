# 整理版本的本地验证

验证日期：2026-11-24。以下检查已实际执行；没有重新运行六组正式训练来替换历史参考结果。

| 检查 | 结果 |
|---|---|
| 可编辑包安装与 `ship-detect` 入口 | 通过 |
| Ruff 代码检查、格式检查 | 通过 |
| CPU 单元检查 | 4 项通过 |
| 重新生成数据审计与划分 | 与原实验摘要完全一致 |
| YOLO11n 两轮短训练、验证和检查点保存 | 通过 |
| 17 类模型 CPU 单图预测 | 通过 |
| 冻结 CLIP、α=0.25 的 GPU 图文预测 | 通过 |
| 官方预测与自定义完整类别得分预测器比较 | 框、类别、置信度一致 |
| 原实验预测重放 | 六个 run × 五种方法，共 30 项评价完全复现 |

重放比较了 COCO AP、AP50、AP75、固定工作点 P/R 和混淆矩阵。原实验的图片、模型、预测缓存需在本机准备；这些文件不随公开代码发布。

## CPU 检查

```bash
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

检查覆盖：真实框构成的理想预测取得 AP=1；空预测取得 AP=0；识别错误与定位错误的区分；α=0 恢复基线；文字改变具体舰级时仍保留 `other_ship`；模型、图像、标签或提示词变化使预测缓存失效。

## 本机模型检查

```bash
python scripts/check_inference.py --weights /path/to/best.pt --image /path/to/photo.jpg --device 0
```

GitHub Actions 已配置为运行 CPU 单元检查和 Ruff。远端执行结果以仓库 Actions 页面为准。
