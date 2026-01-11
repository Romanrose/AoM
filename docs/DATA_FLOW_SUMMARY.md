# AoM项目数据维度变化流程分析

> 按照代码执行顺序，详细分析数据在各模块间的维度变换。

---

## 📊 数据流总览

### 输入数据结构
```
Batch数据 (batch_size=B):
├── input_ids: [B, 66]              # 51图像token + 15文本token
├── image_features: [B, 51, 2048]   # 51个图像ROI特征，2048维
├── labels: [B, target_len]         # 目标序列标签
└── attention_mask: [B, 66]         # 注意力掩码
```

---

## 一、数据加载阶段

### 1.1 批次数据准备
**位置**: `dataset.py: __getitem__`

```python
def __getitem__(self, index):
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),           # [66]
        'image_features': image_features,                                  # [51, 2048]
        'labels': torch.tensor(labels, dtype=torch.long),                 # [target_len]
        'mask': torch.tensor(mask, dtype=torch.long),                     # [66]
    }
```

**维度变化**:
```
单个样本 → 批次 (B=16)
input_ids: [66] → [B, 66]
image_features: [51, 2048] → [B, 51, 2048]
labels: [target_len] → [B, target_len]
mask: [66] → [B, 66]
```

---

## 二、编码器阶段 (BART Encoder)

### 2.1 多模态特征嵌入
**位置**: `src/model/modules.py: 83-94` (_embed_multi_modal)

```python
# 图像特征嵌入
embedded_images = self.embed_images(image_features)  # Linear(2048→768)
# [B, 51, 2048] → [B, 51, 768]

# 文本token嵌入
embedded = self.embed_tokens(input_ids)
# [B, 66] → [B, 66, 768]

# 多模态融合
for index, value in enumerate(embedded_images):
    if len(value) > 0:
        embedded[index, mask[index]] = value
# 将图像特征嵌入到对应位置
# embedded: [B, 66, 768]
```

**维度变化**:
```
输入:
├── image_features: [B, 51, 2048]
└── input_ids: [B, 66]

输出: embedded: [B, 66, 768]
```

### 2.2 BART编码器输出
**位置**: `src/model/modules.py: 130-149`

```python
# Transformer编码
x = x.transpose(0, 1)  # [B, 66, 768] → [66, B, 768]

for encoder_layer in self.layers:
    x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

encoder_outputs = x.transpose(0, 1)  # [66, B, 768] → [B, 66, 768]
```

**维度变化**:
```
输入: embedded: [B, 66, 768]
输出: encoder_outputs: [B, 66, 768] (最终编码结果)
```

---

## 三、A³M模块 (语义对齐)

### 3.1 名词特征提取
**位置**: `MAESC_model.py: 141-159` (get_noun_embed)

```python
noun_embed = torch.zeros(feature.shape[0], max_noun_num, feature.shape[-1]).to(self.mydevice)
# feature: [B, 66, 768]
# noun_embed: [B, L, 768]  (L = max_noun_num)

for i in range(len(feature)):
    noun_embed[i] = torch.index_select(feature[i], dim=0, index=noun_position[i])
    # 从66个token中提取名词位置的特征
```

**维度变化**:
```
输入:
├── encoder_outputs: [B, 66, 768]
└── noun_mask: [B, 66] (标识名词位置)

输出: noun_embed: [B, L, 768]  (L ≈ 5-10)
```

### 3.2 注意力计算 (mode='cat')
**位置**: `MAESC_model.py: 207-222`

#### 步骤1: 特征扩展
```python
multi_features_rep = encoder_outputs.unsqueeze(2).repeat(1, 1, noun_embed.shape[1], 1)
# [B, 66, 768] → [B, 66, 1, 768] → [B, 66, L, 768]

noun_features_rep = noun_embed.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1, 1)
# [B, L, 768] → [B, 1, L, 768] → [B, 66, L, 768]
```

#### 步骤2: 线性变换
```python
noun_features_rep = self.noun_linear(noun_features_rep)  # W_CA
multi_features_rep = self.multi_linear(multi_features_rep)  # W_H
# 维度不变: [B, 66, L, 768]
```

#### 步骤3: 特征拼接
```python
concat_features = torch.tanh(torch.cat([noun_features_rep, multi_features_rep], dim=-1))
# [B, 66, L, 768] + [B, 66, L, 768] → [B, 66, L, 1536]
```

#### 步骤4: 注意力权重
```python
att = torch.softmax(self.att_linear(concat_features).squeeze(-1), dim=-1)
# concat_features: [B, 66, L, 1536] → att_linear → [B, 66, L, 1]
# softmax → [B, 66, L]
```

#### 步骤5: 方面特征
```python
att_features = torch.matmul(att, noun_embed)
# [B, 66, L] @ [B, L, 768] → [B, 66, 768]
```

#### 步骤6: 融合系数
```python
alpha = torch.sigmoid(self.linear(
    torch.cat([self.alpha_linear1(encoder_outputs),
               self.alpha_linear2(att_features)], dim=-1)))
# encoder_outputs: [B, 66, 768]
# alpha_linear1/2: → [B, 66, 768] each
# concat: [B, 66, 1536]
# linear: [1, 1536] → [B, 66, 1]
# sigmoid → [B, 66, 1]

alpha = alpha.repeat(1, 1, 768)  # 广播
# [B, 66, 1] → [B, 66, 768]
```

#### 步骤7: 最终对齐特征
```python
encoder_outputs = torch.mul(1-alpha, encoder_outputs) + torch.mul(alpha, att_features)
# (1-α) × h_t + α × h_t^A
# [B, 66, 768] = ĥ_t (A³M输出)
```

**完整A³M数据维度流**:
```
输入: encoder_outputs: [B, 66, 768]
      noun_embed: [B, L, 768]

流程:
1. 特征扩展 → [B, 66, L, 768]
2. 线性变换 → [B, 66, L, 768]
3. 特征拼接 → [B, 66, L, 1536]
4. 注意力权重 → [B, 66, L]
5. 方面特征 → [B, 66, 768]
6. 融合系数 → [B, 66, 1] → [B, 66, 768]
7. 最终输出 → [B, 66, 768] = ĥ_t
```

---

## 四、AG-GCN模块 (情感聚合)

### 4.1 特征分离
**位置**: `MAESC_model.py: 249-250`

```python
img_feature = encoder_outputs[:, :51, :]  # [B, 51, 768]
text_feature = encoder_outputs[:, 51:, :]  # [B, 15, 768]
```

**维度变化**:
```
输入: encoder_outputs: [B, 66, 768] = ĥ_t

输出:
├── img_feature: [B, 51, 768]  # 图像特征
└── text_feature: [B, 15, 768]  # 文本特征
```

### 4.2 依赖矩阵构建
**位置**: `MAESC_model.py: 248`

```python
new_dependency_matrix = torch.zeros([B, 66, 66], dtype=torch.float).to(encoder_outputs.device)
# [B, 66, 66] (分块矩阵D)
```

### 4.3 文本-文本依赖
**位置**: `MAESC_model.py: 281-284`

```python
text_feature_extend1 = text_feature.unsqueeze(1).repeat(1, 15, 1, 1)
text_feature_extend2 = text_feature.unsqueeze(2).repeat(1, 1, 15, 1)
text_sim = torch.cosine_similarity(text_feature_extend1, text_feature_extend2, dim=-1)
# [B, 15, 768] → [B, 15, 15, 768] (扩展)
# 余弦相似度 → text_sim: [B, 15, 15]

new_dependency_matrix[:, 51:, 51:] = dependency_matrix * text_sim
# [B, 15, 15] (填入D_TT子矩阵)
```

### 4.4 图像-文本依赖
**位置**: `MAESC_model.py: 270-278`

```python
img_feature_extend = img_feature.unsqueeze(2).repeat(1, 1, text_feature.shape[1], 1)
text_feature_extend = text_feature.unsqueeze(1).repeat(1, img_feature.shape[1], 1, 1)
sim = torch.cosine_similarity(img_feature_extend, text_feature_extend, dim=-1)
# [B, 51, 768] & [B, 15, 768] → [B, 51, 15, 768]
# 余弦相似度 → sim: [B, 51, 15]

noun_mask = noun_mask[:, 51:].unsqueeze(1).repeat(1, sim.shape[1], 1)
sim = sim * noun_mask  # 仅保留名词相关

new_dependency_matrix[:, :51, 51:] = sim  # D_VT
new_dependency_matrix[:, 51:, :51] = torch.transpose(sim, 1, 2)  # D_TV
# [B, 51, 15] & [B, 15, 51]
```

**依赖矩阵维度变化**:
```
输入:
├── text_feature: [B, 15, 768]
└── img_feature: [B, 51, 768]

文本依赖:
text_sim: [B, 15, 15] → D_TT: [B, 15, 15]

图像依赖:
sim: [B, 51, 15] → D_VT: [B, 51, 15]
                     D_TV: [B, 15, 51]

最终: D: [B, 66, 66] = [[D_VV, D_VT], [D_TV, D_TT]]
```

### 4.5 情感特征处理
**位置**: `MAESC_model.py: 294-298`

```python
sentiment_value = nn.ZeroPad2d(padding=(51, 0, 0, 0))(sentiment_value)
# [B, 15] → [B, 66]

sentiment_value = sentiment_value.unsqueeze(-1)
# [B, 66] → [B, 66, 1]

sentiment_feature = self.senti_value_linear(sentiment_value)
# [B, 66, 1] → senti_value_linear → [B, 66, 768] = s_i
```

**维度变化**:
```
输入: sentiment_value: [B, 15] (SenticNet分数)

流程:
1. ZeroPad2d → [B, 66]
2. unsqueeze → [B, 66, 1]
3. Linear → [B, 66, 768] = s_i
```

### 4.6 情感-语义融合
**位置**: `MAESC_model.py: 298`

```python
context_feature = self.context_linear(encoder_outputs + sentiment_feature)
# [B, 66, 768] + [B, 66, 768] → [B, 66, 768]
# Linear → [B, 66, 768] = h_i^S
```

### 4.7 图卷积计算 (GCN)
**位置**: `MAESC_model.py: 299` + `GCN.py`

```python
context_feature = self.context_gcn(context_feature, context_dependency_matrix, attention_mask)

# GCN内部:
def forward(self, inputs, adj, mask=None):
    # inputs: [B, 66, 768] = h_{i,l-1}^S
    # adj: [B, 66, 66] = A (关联矩阵)

    support = torch.matmul(inputs, self.weight)
    # [B, 66, 768] @ [768, 768] → [B, 66, 768]

    output = torch.matmul(adj, support)
    # [B, 66, 66] @ [B, 66, 768] → [B, 66, 768]

    output = self.act(output + self.bias)  # ReLU
    return output
```

**维度变化**:
```
输入:
├── context_feature: [B, 66, 768] = h_{i,l-1}^S
└── context_dependency_matrix: [B, 66, 66] = A

GCN计算:
support: [B, 66, 768]
→ output: [B, 66, 768] = h_{i,l}^S

输出: GCN_output: [B, 66, 768] = Ĥ^S
```

### 4.8 最终融合
**位置**: `MAESC_model.py: 302`

```python
mix_feature = self.gcn_proportion * context_feature + encoder_outputs
# gcn_proportion = 0.5 (默认)
# 0.5 × Ĥ^S + 1.0 × Ĥ = H̃
```

**维度变化**:
```
输入:
├── context_feature (GCN): [B, 66, 768] = Ĥ^S
└── encoder_outputs: [B, 66, 768] = Ĥ

融合: 0.5 × [B, 66, 768] + 1.0 × [B, 66, 768] → [B, 66, 768]

输出: mix_feature: [B, 66, 768] = H̃ (AG-GCN输出)
```

---

## 五、解码器阶段 (BART Decoder)

### 5.1 解码器前向传播
**位置**: `src/model/modules.py: 237-243`

```python
dict = self.decoder(
    input_ids=tokens,
    encoder_hidden_states=mix_feature,  # [B, 66, 768]
    encoder_padding_mask=encoder_pad_mask,
    decoder_padding_mask=decoder_pad_mask,
    decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
    return_dict=True)

hidden_state = dict.last_hidden_state
# [B, tgt_len, 768] = h_t^d
```

**维度变化**:
```
输入:
├── tokens: [B, tgt_len] (目标序列)
├── mix_feature (encoder输出): [B, 66, 768] = H̃
├── encoder_pad_mask: [B, 66]
└── decoder_pad_mask: [B, tgt_len]

BART Decoder输出:
hidden_state: [B, tgt_len, 768] = h_t^d
```

---

## 六、预测输出阶段

### 6.1 情感分类
**位置**: `src/model/modules.py: 273-278`

```python
tag_scores = F.linear(
    hidden_state,
    self.dropout_layer(
        self.decoder.embed_tokens.weight[self.label_start_id:self.label_start_id + 3]))
# hidden_state: [B, tgt_len, 768]
# embed_tokens.weight[label]: [3, 768] (POS, NEU, NEG)
# F.linear: [B, tgt_len, 768] @ [3, 768]^T → [B, tgt_len, 3]

logits[:, :, 3:self.src_start_index] = tag_scores
# logits: [B, tgt_len, num_classes] = P(y_t)
```

**维度变化**:
```
输入:
├── hidden_state: [B, tgt_len, 768] = h_t^d
└── label_embeddings: [3, 768] (POS, NEU, NEG)

计算: [B, tgt_len, 768] @ [3, 768]^T → [B, tgt_len, 3]

输出: logits: [B, tgt_len, num_classes] = P(y_t)
     其中情感类别: POS(0), NEU(1), NEG(2)
```

---

## 七、完整数据流总结

### 完整维度变化链

```
输入批次 (B=16):
├── input_ids: [B, 66]
├── image_features: [B, 51, 2048]
└── labels: [B, tgt_len]

    ↓ [编码器]
encoder_outputs: [B, 66, 768]

    ↓ [A³M模块]
noun_embed: [B, L, 768]
→ att: [B, 66, L]
→ att_features: [B, 66, 768]
→ alpha: [B, 66, 768]
→ ĥ_t: [B, 66, 768]

    ↓ [AG-GCN模块]
img_feature: [B, 51, 768]
text_feature: [B, 15, 768]
→ sim: [B, 51, 15]
→ dependency_matrix: [B, 66, 66]
→ sentiment_feature: [B, 66, 768]
→ GCN_output: [B, 66, 768] = Ĥ^S
→ H̃: [B, 66, 768]

    ↓ [解码器]
hidden_state: [B, tgt_len, 768] = h_t^d

    ↓ [预测输出]
logits: [B, tgt_len, num_classes] = P(y_t)
```

### 关键维度常数

| 维度名称 | 值 | 说明 |
|----------|----|----|
| B | 16 | 批次大小 |
| seq_len | 66 | 输入序列总长度 |
| img_len | 51 | 图像token数量 |
| text_len | 15 | 文本token数量 |
| img_feat_dim | 2048 | 图像ROI特征维度 |
| hidden_dim | 768 | 文本/隐藏层维度 |
| noun_len | L | 名词数量 (动态) |
| tgt_len | 10 | 目标序列长度 |
| num_classes | 50265 | 输出类别数 |

### 关键张量操作

| 操作 | 输入维度 | 输出维度 | 说明 |
|------|----------|----------|------|
| `unsqueeze` | `[B, N]` | `[B, 1, N]` | 扩展维度 |
| `repeat` | `[B, N]` | `[B, L, N]` | 重复数据 |
| `matmul` | `[B, N, M] @ [B, M, K]` | `[B, N, K]` | 批量矩阵乘 |
| `cosine_similarity` | `[B, N, D], [B, M, D]` | `[B, N, M]` | 余弦相似度 |
| `torch.cat` | `[B, N, D], [B, N, D]` | `[B, N, 2D]` | 维度拼接 |
| `Linear` | `[B, N, D_in]` | `[B, N, D_out]` | 线性变换 |
| `softmax` | `[B, N]` | `[B, N]` | 归一化 |

---

**分析完成时间**: 2025-11-13
**数据流完整性**: ✅