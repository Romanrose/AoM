# 分词过程演示文档

## 概述

本演示展示了 `tokenization_new.py` 中的 `encode_condition` 函数如何处理多模态输入（文本 + 图像）并生成模型所需的编码结果。

## 输入示例

- **句子**: "I love this phone"
- **图像特征数量**: 51个ROI特征
- **任务**: 方面级情感分类 (AESC)

## 分词流程详解

### 1. 输入准备

```python
sentence = "I love this phone"
img_num = 51  # 图像特征数量
noun_list = ['NNP', 'NNPS', 'NN', 'NNS']  # 名词POS标签列表
```

### 2. 文本分词

```python
sentence_split = ['I', 'love', 'this', 'phone']
```

### 3. POS标注与名词识别

使用 spaCy 进行词性标注：

| 词 | POS | 是否名词 | 说明 |
|----|-----|----------|------|
| I | PRP | False | 人称代词 |
| love | VBP | False | 动词 |
| this | DT | False | 限定词 |
| phone | **NN** | **True** | **名词** |

**名词位置索引**: `[3]` (phone在索引3处)

### 4. BPE分词

使用 BART 分词器进行 BPE 分词：

| 词 | BPE Tokens | 是否名词 | 名词掩码 |
|----|-----------|----------|----------|
| I | [100] | False | [0] |
| love | [200, 201] | False | [0, 0] |
| this | [300] | False | [0] |
| phone | [400, 401] | **True** | **[1, 1]** |

**最终文本序列**:
```python
# BOS token + 文本tokens + EOS token
[50276, 100, 200, 201, 300, 400, 401, 50277]
```

**最终名词掩码**:
```python
[0, 0, 0, 0, 0, 1, 1, 0]  # phone的两个BPE token被标记为1
```

### 5. 图像特征编码

```python
# 图像token序列格式
<<img>> <<img_feat>> * 51 <<img>>
```

**编码后的图像ID序列**:
```python
[50273, 50273, ..., 50273]  # 51个50273 (img_feat_id)
```

### 6. 图像与文本拼接

```python
final_input_ids = image_ids + word_bpes
# [51个图像特征] + [8个文本tokens] = 59个tokens
```

**最终输入序列**:
```
[50273, 50273, ..., 50273, 50276, 100, 200, 201, 300, 400, 401, 50277]
 ^--------------------^  ^-------------------------------------------^
      51个图像特征                    8个文本tokens
```

### 7. 掩码生成

#### 7.1 注意力掩码

```python
attention_mask = [1] * 59  # 所有位置都参与注意力计算
```

#### 7.2 名词掩码

```python
noun_mask = [0]*51 + [0, 0, 0, 0, 0, 1, 1, 0]
# 前51个(图像)  后8个(文本)
```

**名词位置**: 索引56和57 (phone的两个BPE token)

#### 7.3 图像掩码

```python
img_mask = [True]*51 + [False]*8
```

#### 7.4 句子掩码

```python
sentence_mask = [False]*51 + [True]*8
```

### 8. 依存关系矩阵 (GCN)

**矩阵维度**: 8×8 (仅文本部分)

**依存关系**:

| 位置 | 词 | 依存关系 |
|------|----|--------|
| 0 | BOS | 自连接 |
| 1 | I | love → I (主谓) |
| 2 | love | 自连接; I → love; phone → love |
| 3 | this | 自连接; phone → this |
| 4 | this | 自连接; love → this; phone → this |
| 5 | phone | 自连接; love → phone (动宾); this → phone |
| 6 | phone | 自连接 |
| 7 | EOS | 自连接 |

**矩阵表示**:

```
     0  1  2  3  4  5  6  7
0 [ 5, 0, 0, 0, 0, 0, 0, 0]
1 [ 1, 5, 0, 0, 1, 0, 0, 0]  # love -> I, phone
2 [ 0, 0, 5, 0, 1, 0, 0, 0]  # phone -> this
3 [ 0, 0, 0, 5, 0, 0, 0, 0]
4 [ 0, 0, 0, 0, 5, 0, 0, 0]
5 [ 0, 0, 0, 0, 0, 5, 0, 0]
6 [ 0, 0, 0, 0, 0, 0, 5, 0]
7 [ 0, 0, 0, 0, 0, 0, 0, 5]
```

**权重规则**:
- 对角线: 5 (自连接)
- 直接依存关系: 1 (主谓、动宾、修饰)
- 间接依存关系: 1 (祖孙关系)

### 9. SenticNet情感值 (可选)

如果启用 SenticNet，则为每个token分配情感分数：

```python
sentiment_value = [0.0] * 59  # 所有非名词词为0
```

## AESC标签编码

### AESC格式

```
[<<text>>] [text words] [<</text>>] [AESC] [start] [end] [polarity] [EOS]
```

### 示例

**输入**:
- 句子: "I love this phone"
- 方面-情感: [(1, 3, 'POS')]  # love, POS

**编码**:
1. 添加文本标记: `<<text>>`
2. 添加文本: `I love this phone`
3. 添加文本结束: `<</text>>`
4. 添加AESC标记: `<<AESC>>`
5. 添加方面位置: start=?, end=?
6. 添加情感极性: polarity=?

**最终输出**:
```python
[1000, 100, 200, 300, 400, 1001, 5000, 1002, 1005, 3, 50277]
```

| 位置 | Token | ID | 说明 |
|------|-------|----|----|
| 0 | `<<text>>` | 1000 | 文本开始 |
| 1-4 | text | 100, 200, 300, 400 | 文本内容 |
| 5 | `<</text>>` | 1001 | 文本结束 |
| 6 | `<<AESC>>` | 5000 | AESC任务标记 |
| 7 | start | 1002 | 方面起始位置 |
| 8 | end | 1005 | 方面结束位置 |
| 9 | polarity | 3 | 情感极性ID (POS) |
| 10 | EOS | 50277 | 序列结束 |

## 最终编码结果

### 输出字典

```python
{
    'input_ids': [59],           # 输入token ID序列
    'attention_mask': [59],        # 注意力掩码 (全1)
    'noun_mask': [59],           # 名词位置掩码
    'dependency_matrix': [8x8],   # 依存关系矩阵 (仅文本)
    'sentiment_value': [59],      # SenticNet情感值
    'img_mask': [59],            # 图像位置掩码
    'sentence_mask': [59]         # 句子位置掩码
}
```

### 关键统计

| 指标 | 值 |
|------|----|
| 输入序列总长度 | 59 |
| 图像特征数量 | 51 |
| 文本token数量 | 8 |
| 名词数量 | 2 |
| 依存矩阵维度 | 8×8 |

## 运行演示

```bash
cd /home/ljy/data/media/projects/aom
python3 tests/test_tokenization_demo.py
```

## 关键特性总结

### 1. 多模态融合
- 图像特征与文本token联合编码
- 固定长度: 51 (图像) + 15 (文本) = 66

### 2. 方面感知
- POS标注识别名词
- 方面span标记 (起始、结束位置)
- 情感极性联合预测

### 3. 情感知识增强
- SenticNet情感词典注入
- 每个token的情感分数

### 4. 依存关系建模
- spaCy依存句法分析
- GCN友好的矩阵格式
- BPE扩展保持依存关系

### 5. 特殊任务Token
- 18个特殊token标记不同任务
- 支持方面提取、情感分类、联合任务

---

**注**: 本演示基于模拟数据，实际运行需要加载真实的BART分词器和spaCy模型。
