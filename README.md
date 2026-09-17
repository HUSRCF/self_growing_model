# Self-growing model：搜索与事件动力学研究

`master` 用于仓库入口，`shuang` 保存当前研究代码与判断。

## 当前结论

- 原型的中心差分检查器接口已修正，并有针对性测试。
- 0.5 秒开发集误差：固定深度2为0.035752，可变深度2–3为0.035754。
- 复用测试集误差分别为0.092618与0.092649；64个窗口均完成。
- 扩展死节点的候选范围没有改善最终预测；简单软化检查反而略差。
- 首个小型价值回归模型泛化不足，未启用于正式搜索。
- 下一步：增加生成状态覆盖，训练同前缀候选排序，按整视频验证选路收益。

详细依据：[修正后实验记录](.agents/notes/corrected_search_experiments.md)。历史记录中的错误接口结果仅供追溯，不能视为当前模型效果。

## 本次提交范围与复现

提交研究脚本、测试、JSON结果摘要与`.agents/`长期记忆；不提交原始ZIP、原始视频数据、冻结版本包及大型预测数组。

运行前需将原始`v20_rnn_mixture_complete.zip`解压为本目录下的`v20_rnn_mixture/`，其中须包含`engine/`、`models/`、`data/`。归档SHA-256为`8a9a8099bc9aff664d77437d79ffc0fe9a99e21864a39a978054c414004ce721`。Python依赖见该包的`requirements-predict.txt`。

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest -v test_adaptive_search
OPENBLAS_NUM_THREADS=1 python adaptive_search_prototype.py --per-video 8 --particles 2 --steps 50 --min-depth 2 --max-depth 3 --output adaptive_search_results/reproduced_dev.json
OPENBLAS_NUM_THREADS=1 python train_search_value.py
```

当前温度0的重复粒子没有概率多样性；结果不应与原版8随机粒子、300步的窗口结果直接比较。数据划分与窗口限制见`.agents/`记录。
