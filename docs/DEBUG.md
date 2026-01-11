# AoM项目调试指南

> 本文档提供迁移后AoM项目的完整调试手册，包括常见错误、解决方案和调试技巧。

## 📋 目录

1. [快速诊断清单](#快速诊断清单)
2. [环境问题排查](#环境问题排查)
3. [路径问题排查](#路径问题排查)
4. [依赖问题排查](#依赖问题排查)
5. [运行时错误排查](#运行时错误排查)
6. [性能问题排查](#性能问题排查)
7. [调试工具和技巧](#调试工具和技巧)
8. [常见错误解决方案](#常见错误解决方案)
9. [日志分析方法](#日志分析方法)

---

## 快速诊断清单

### ✅ 基础检查

```bash
# 1. 检查项目结构
cd /home/ljy/data/media/projects/aom
ls -la
# 应包含: src/, configs/, checkpoints/, logs/, records/, results/, scripts/

# 2. 检查环境
conda activate aom
python --version  # 应为 3.8.x
torch.__version__  # 应为 1.6.0

# 3. 检查关键文件
ls -la run_aom.py  # 启动脚本
ls -la global_var.py  # 全局配置
ls -la MAESC_training.py  # 训练脚本

# 4. 测试初始化
python run_aom.py --task twitter15 --no_train
# 如看到初始化日志且无错误，则基础配置正确
```

### ⚠️ 常见警告

```
UserWarning: Torched binary built with Volta too old for this GPU
```
**状态**: 非致命警告
**影响**: RTX 4090将使用CPU模式训练
**解决**: 可升级PyTorch到1.12.0+CUDA11.3（可选）

---

## 环境问题排查

### 1. Conda环境问题

#### 问题1: 环境不存在
```bash
# 现象
CommandNotFoundError: activate: No such file or directory.

# 解决
conda env list | grep aom  # 查找环境
# 如果不存在，从环境文件创建
conda env create -f configs/environment.yaml -n aom
conda activate aom
```

#### 问题2: 环境路径错误
```bash
# 现象
ModuleNotFoundError: No module named 'torch'

# 诊断
which python
which conda
# 应在 /home/ljy/miniconda3/envs/aom/bin/python

# 解决
conda activate aom
# 或指定完整路径
/home/ljy/miniconda3/envs/aom/bin/python run_aom.py
```

### 2. Python版本兼容性

#### 检查版本
```python
import sys
print(sys.version)
# 应输出: 3.8.x

import torch
print(torch.__version__)
# 应输出: 1.6.0 或兼容版本
```

#### 常见版本问题

**spaCy 2.x vs 3.x**
```python
# 症状1
ModuleNotFoundError: No module named 'spacy.lang.en.tag_map'

# 解决（已修复）
# 代码中已添加兼容性处理
try:
    from spacy.lang.en.tag_map import TAG_MAP
except ImportError:
    from spacy.lang.en import TAG_MAP

# 症状2
# spaCy 3.x需要单独下载模型
python -m spacy download en_core_web_sm
```

**transformers版本**
```python
# 检查
import transformers
print(transformers.__version__)
# 应为 3.0.2

# 症状
NameError: name 'Seq2SeqModelOutput' is not defined

# 解决（已修复）
# modeling_bart.py中已添加自定义输出类型定义
```

---

## 路径问题排查

### 1. 验证路径配置

#### 检查global_var.py
```bash
# 文件路径: /home/ljy/data/media/projects/aom/global_var.py
# 应包含正确的绝对路径
grep -E "(twitter15|twitter17|TRC)_data_dir" global_var.py
```

#### 检查run_aom.py路径
```bash
# 验证日志和检查点路径设置
python run_aom.py --task twitter15 --no_train 2>&1 | grep -E "(checkpoint_dir|log_dir)"
# 应输出:
# checkpoint_dir: ./train15
# log_dir: 15_aesc
```

### 2. 关键路径清单

| 文件 | 路径 | 说明 |
|------|------|------|
| SenticNet | `/home/ljy/data/media/projects/aom/src/senticnet_word.txt` | 情感知识库 |
| 数据集JSON | `/home/ljy/data/media/projects/aom/src/data/jsons/` | 数据配置文件 |
| BART模型 | `/home/ljy/data/media/projects/aom/src/model/bart-base` | 预训练模型 |
| TRC检查点 | `/home/ljy/data/media/projects/aom/checkpoints/pytorch_model.bin` | 预训练权重 |
| 训练输出 | `./train15/` 或 `./train17/` | 训练检查点 |
| 日志输出 | `15_aesc/` 或 `17_aesc/` | TensorBoard日志 |

### 3. 路径验证命令

```bash
# 检查所有关键路径
cd /home/ljy/data/media/projects/aom

# 数据文件
ls -la src/senticnet_word.txt
ls -la src/data/jsons/twitter15_info.json
ls -la src/data/jsons/twitter17_info.json

# 模型文件
ls -la src/model/bart-base/  # 目录存在即可
ls -la checkpoints/pytorch_model.bin

# 脚本文件
ls -la run_aom.py
ls -la MAESC_training.py
```

### 4. 常见路径错误

#### 错误1: FileNotFoundError
```
FileNotFoundError: [Errno 2] No such file or directory: 'AoM-ckpt/Twitter2015/AoM.pt'
```
**原因**: 测试模式缺少模型文件
**解决**: 使用`--model_path`指定模型路径，或先训练模型

#### 错误2: 相对路径错误
```
FileNotFoundError: [Errno 2] No such file or directory: '../../senticnet_word.txt'
```
**原因**: tokenization_new.py路径计算错误
**解决**: 已修复，使用相对路径`../senticnet_word.txt`

---

## 依赖问题排查

### 1. 关键依赖检查

```bash
# 检查所有依赖
pip list | grep -E "(torch|transformers|spacy|numpy|fastNLP)"

# 版本信息
pip show torch transformers spacy
```

### 2. 缺失依赖

#### 错误: ModuleNotFoundError
```bash
# 症状
ModuleNotFoundError: No module named 'fastNLP'
ModuleNotFoundError: No module named 'timm'

# 解决
pip install fastNLP==0.7.0 timm==0.6.7
```

#### pytorch-transformers兼容性
```python
# 错误
ModuleNotFoundError: No module named 'pytorch_transformers'

# 解决（已修复）
# MAESC_training.py line 8-10已添加
import sys
sys.modules["pytorch_transformers"] = __import__("transformers")
```

### 3. 版本冲突

#### pydantic版本
```bash
# 症状
ImportError: cannot import name 'BaseModel' from 'pydantic'

# 解决
pip install pydantic==1.8.2
```

#### typing-extensions
```bash
# 症状
ImportError: cannot import name 'Literal' from 'typing_extensions'

# 解决
pip install typing-extensions==3.10.0.0
```

---

## 运行时错误排查

### 1. 初始化错误

#### Tokenizer初始化失败
```
NameError: name 'os' is not defined
```
**位置**: `src/data/tokenization_new.py:136`
**原因**: 缺少os模块导入
**解决**: 已在文件头部添加`import os`

#### SenticNet读取失败
```
FileNotFoundError: [Errno 2] No such file or directory: '...senticnet_word.txt'
```
**诊断**:
```python
# 验证文件存在
ls -la /home/ljy/data/media/projects/aom/src/senticnet_word.txt
# 文件大小应为 618074 字节
```

### 2. 训练启动失败

#### 重复参数错误
```
# 症状
error: argument --checkpoint_dir: expected one argument
```
**原因**: 命令行中包含两个`--checkpoint_dir`
**解决**: 已修复run_aom.py，移除重复参数

#### GPU初始化失败
```
CUDA out of memory. Trying to allocate 0.00 MiB
```
**诊断**:
```python
import torch
print(torch.cuda.is_available())  # False for RTX 4090 + PyTorch 1.6.0
print(torch.cuda.get_device_name(0))  # None
```
**解决**: 继续使用CPU训练，或升级PyTorch

### 3. 数据加载问题

#### 内存不足
```
RuntimeError: Unable to find a valid socket to bind to
```
**原因**: 多进程数据加载与CUDA设置冲突
**解决**: 设置`--num_workers 0`（已在默认配置中）

---

## 性能问题排查

### 1. CPU vs GPU性能

#### 检查计算设备
```python
# 训练脚本会自动检测
device: gpu  # 实际运行在CPU
```
**原因**: PyTorch 1.6.0不支持RTX 4090架构

#### 性能对比
| 设备 | 速度 | 内存 | 兼容 |
|------|------|------|------|
| CPU (38核心) | 慢 | 无限制 | ✅ |
| GPU (RTX 4090) | 快 | 24GB | ❌ PyTorch 1.6.0 |

### 2. 内存使用监控

```bash
# 监控内存使用
watch -n 1 free -h

# Python内存使用
ps aux | grep MAESC_training
```

### 3. I/O瓶颈

#### 数据加载慢
```python
# 检查点: 使用SSD存储数据集
# 解决: 确保数据集在本地SSD而非网络存储
```

---

## 调试工具和技巧

### 1. Python调试器

#### 使用pdb
```python
# 在代码中设置断点
import pdb; pdb.set_trace()

# 然后继续执行
# c: continue
# s: step
# n: next
# l: list
# q: quit
```

#### 使用ipdb（推荐）
```bash
pip install ipdb
# 在代码中使用
import ipdb; ipdb.set_trace()
```

### 2. 日志调试

#### 启用详细日志
```bash
# 设置环境变量
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1

# 运行训练
python run_aom.py --task twitter15 2>&1 | tee train.log
```

#### 查看特定日志
```bash
# 实时监控日志
tail -f logs/train15/log.txt

# 搜索错误
grep -i error logs/*/log.txt

# 搜索损失值
grep -E "Epoch.*Loss" logs/*/log.txt
```

### 3. 可视化调试

#### TensorBoard
```bash
# 启动TensorBoard
tensorboard --logdir=15_aesc/  # 或 17_aesc/

# 浏览器访问
# http://localhost:6006
```

#### 检查模型
```python
# 在训练脚本中
print(model)
print(model.parameters())
```

### 4. 性能分析

#### 使用cProfile
```bash
python -m cProfile -o profile.stats MAESC_training.py [args]
# 分析结果
python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('time').print_stats(20)"
```

---

## 常见错误解决方案

### 1. 环境问题

#### 错误: `python3.7: not found`
```bash
# 症状
/usr/bin/python3.7: No such file or directory

# 解决（已修复）
# 脚本中已更改 python3.7 为 python
```

### 2. 依赖问题

#### 错误: `ModuleNotFoundError: No module named 'spacy'`
```bash
# 解决
conda activate aom
pip install spacy==2.1.4
python -m spacy download en_core_web_sm
```

#### 错误: `ModuleNotFoundError: No module named 'fastNLP'`
```bash
# 解决
pip install fastNLP==0.7.0 timm==0.6.7
```

### 3. 路径问题

#### 错误: `FileNotFoundError` for senticnet_word.txt
```python
# 检查文件存在
import os
path = '/home/ljy/data/media/projects/aom/src/senticnet_word.txt'
assert os.path.exists(path), f"File not found: {path}"
print(f"File size: {os.path.getsize(path)} bytes")
```

### 4. 内存问题

#### 错误: `RuntimeError: CUDA out of memory`
```python
# 方案1: 减少batch_size
python run_aom.py --task twitter15 --batch_size 8

# 方案2: 使用CPU（RTX 4090 + PyTorch 1.6.0 自动回退）
# 无需修改，已自动处理
```

### 5. 训练问题

#### 错误: `nan loss`
```python
# 检查学习率
--lr 7.5e-5  # 过大或过小都可能导致

# 检查梯度裁剪
--grad_clip 5.0

# 检查数据
# 确保数据格式正确，无异常值
```

---

## 日志分析方法

### 1. 训练日志结构

```
📁 项目根目录/
├── 📁 15_aesc/          # Twitter15训练日志
│   ├── 📄 log.txt       # 文本日志
│   └── 📁 events.out.tfevents/  # TensorBoard事件文件
├── 📁 train15/          # 检查点目录
│   ├── 📄 pytorch_model.bin
│   └── 📁 model_*/      # 每个epoch的模型
```

### 2. 关键日志指标

#### 训练过程
```
Epoch [1/35], Step [100/4620], Loss: 4.6640
```
**含义**:
- `Epoch [1/35]`: 第1轮/共35轮
- `Step [100/4620]`: 第100步/共4620步
- `Loss: 4.6640`: 当前损失值

#### 验证过程
```
Eval: Epoch [5], AESC F1: 0.6842, AE F1: 0.8245, SC F1: 0.7321
```
**含义**:
- `AESC F1`: 方面情感分类联合任务F1分数
- `AE F1`: 方面抽取F1分数
- `SC F1`: 情感分类F1分数

### 3. 问题诊断日志

#### 内存泄漏
```
RuntimeWarning: CUDA memory usage increasing continuously
```
**诊断**:
```python
# 在训练循环中监控内存
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

#### 数据加载慢
```
RuntimeWarning: DataLoader worker (pid) is killed by the OOM killer
```
**解决**: 设置`--num_workers 0`

---

## 调试流程图

```
┌─────────────────┐
│   运行失败       │
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  1. 检查环境     │
│  conda activate │
│  python --ver   │
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  2. 检查路径     │
│  ls 关键文件     │
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  3. 检查依赖     │
│  pip list       │
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  4. 运行测试     │
│  --no_train     │
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  5. 查看日志     │
│  grep -i error  │
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  问题定位        │
│  使用pdb/ipdb   │
└─────────────────┘
```

---

## 常用调试命令速查

```bash
# 环境检查
conda activate aom && python --version && python -c "import torch; print(torch.__version__)"

# 路径检查
ls -la src/senticnet_word.txt
ls -la src/data/jsons/twitter*_info.json
ls -la checkpoints/pytorch_model.bin

# 依赖检查
pip list | grep -E "torch|transformers|spacy|fastNLP"

# 运行测试
python run_aom.py --task twitter15 --no_train 2>&1 | tee debug.log

# 搜索错误
grep -i "error\|exception\|traceback" debug.log

# 监控日志
tail -f 15_aesc/log.txt

# 内存使用
watch -n 1 'free -h; ps aux | grep MAESC_training | awk "{print \$6}"'

# GPU状态（如果可用）
nvidia-smi  # 或
watch -n 1 nvidia-smi
```

---

## 联系与支持

如果遇到本文档未涵盖的问题：

1. **检查日志**: `logs/*/log.txt`
2. **检查错误输出**: 运行命令并查看stderr
3. **搜索类似问题**: 检查GitHub Issues
4. **环境复现**: 使用`conda env export`保存环境
5. **调试信息收集**:
   ```bash
   python -c "import sys, torch, transformers; print(sys.version, torch.__version__, transformers.__version__)"
   ```

---

**最后更新**: 2025-11-13
**版本**: v1.0