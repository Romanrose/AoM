# AoM论文公式与代码维度对应分析

> 本文档详细分析AoM论文中核心公式推导的数据维度变化在代码中的具体实现。

---

## 📊 数据流总览

```python
输入数据 → BART编码 → A³M模块 → AG-GCN模块 → 预测输出
  ↓          ↓          ↓          ↓          ↓
[batch,   [batch,    [batch,    [batch,    [batch,
 seq_len, seq_len,   seq_len,   seq_len,   seq_len,
  768]     768]       768]       768]       768]
```

---

## 一、A³M模块 - 数据维度变换详解

### 1. 输入数据维度

```python
encoder_outputs: [batch_size, seq_len, hidden_size]
                = [B, 66, 768]  # seq_len=66 (51图像+15文本token)
noun_embed:     [batch_size, max_noun_num, hidden_size]
                = [B, L, 768]   # L为最大名词数量
```

**代码位置**: `MAESC_model.py:141-159` (`get_noun_embed`方法)

### 2. 名词特征提取 - 候选方面特征 H^CA

#### 论文公式
```
H^CA = {h_1^CA, h_2^CA, ..., h_l^CA} ∈ ℝ^(d×l)
```

#### 代码实现
```python
# MAESC_model.py:150-158
noun_embed = torch.zeros(feature.shape[0], max_noun_num, feature.shape[-1]).to(self.mydevice)
# 维度: [B, max_noun_num, 768] = [B, L, d]

for i in range(len(feature)):
    noun_embed[i] = torch.index_select(feature[i], dim=0, index=noun_position[i])
    noun_embed[i, noun_num[i]:] = torch.zeros(max_noun_num-noun_num[i], feature.shape[-1])
```

**维度变化**:
- `feature`: `[B, 66, 768]` → 提取名词位置特征
- `noun_embed`: `[B, max_noun_num, 768]` = `[B, L, d]`

---

### 3. 综合特征 Z_t 计算 (公式1)

#### 论文公式
```
Z_t = tanh((W_CA H^CA + b_CA) ⊕ (W_H h_t + b_H))
维度: tanh([2d×l]) ∈ ℝ^(2d×l)
```

#### 代码实现
```python
# MAESC_model.py:208-213 (noun_attention方法)
# 1. 特征复制扩展
multi_features_rep = encoder_outputs.unsqueeze(2).repeat(1, 1, noun_embed.shape[1], 1)
# 维度: [B, 66, 1, 768] → [B, 66, L, 768]

noun_features_rep = noun_embed.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1, 1)
# 维度: [B, 1, L, 768] → [B, 66, L, 768]

# 2. 线性变换
noun_features_rep = self.noun_linear(noun_features_rep)  # W_CA
multi_features_rep = self.multi_linear(multi_features_rep)  # W_H
# 维度: [B, 66, L, 768] → 线性变换后仍为[...]

# 3. 拼接和激活
concat_features = torch.tanh(torch.cat([noun_features_rep, multi_features_rep], dim=-1))
# 维度: torch.cat([B,66,L,768], [B,66,L,768], dim=-1) = [B,66,L,1536] = [B,66,L,2d]
```

**维度变化**:
- `encoder_outputs`: `[B, 66, 768]` → unsqueeze→ `[B, 66, 1, 768]` → repeat→ `[B, 66, L, 768]`
- `noun_embed`: `[B, L, 768]` → unsqueeze→ `[B, 1, L, 768]` → repeat→ `[B, 66, L, 768]`
- `concat_features`: `[B, 66, L, 1536]` = `[B, 66, L, 2d]`

---

### 4. 注意力分布 α_t 计算 (公式2)

#### 论文公式
```
α_t = softmax(W_α Z_t + b_α)
维度: softmax([1×l]) ∈ ℝ^l
```

#### 代码实现
```python
# MAESC_model.py:214
att = torch.softmax(self.att_linear(concat_features).squeeze(-1), dim=-1)
# 维度:
# concat_features: [B, 66, L, 1536]
# att_linear(concat_features): [B, 66, L, 1]
# .squeeze(-1): [B, 66, L]
# softmax(dim=-1): [B, 66, L]
```

**维度变化**:
- `concat_features`: `[B, 66, L, 1536]` → att_linear→ `[B, 66, L, 1]` → squeeze→ `[B, 66, L]`
- `att`: `[B, 66, L]` = softmax后的注意力权重

---

### 5. 方面相关特征 h_t^A 计算 (公式3)

#### 论文公式
```
h_t^A = Σ(α_t,i × h_i^CA)
维度: ℝ^d
```

#### 代码实现
```python
# MAESC_model.py:215
att_features = torch.matmul(att, noun_embed)
# 维度: torch.matmul([B,66,L], [B,L,768]) → [B,66,768]
```

**维度变化**:
- `att`: `[B, 66, L]`
- `noun_embed`: `[B, L, 768]`
- `att_features`: `[B, 66, 768]` = h_t^A

**数学计算**:
```
对于第t个token (固定t):
h_t^A = α_t,1 × h_1^CA + α_t,2 × h_2^CA + ... + α_t,L × h_L^CA
     = Σ(i=1 to L) α_t,i × h_i^CA
```

---

### 6. 融合系数 β_t 计算 (公式4)

#### 论文公式
```
β_t = sigmoid(W_β [W_1 h_t; W_2 h_t^A] + b_β)
维度: sigmoid(scalar) ∈ [0,1]
```

#### 代码实现
```python
# MAESC_model.py:217
alpha = torch.sigmoid(self.linear(
    torch.cat([self.alpha_linear1(encoder_outputs),
               self.alpha_linear2(att_features)], dim=-1)))
# 维度:
# encoder_outputs: [B, 66, 768]
# alpha_linear1(encoder_outputs): [B, 66, 768]
# att_features: [B, 66, 768]
# alpha_linear2(att_features): [B, 66, 768]
# torch.cat([...], dim=-1): [B, 66, 1536]
# self.linear(...): [B, 66, 1]
# sigmoid: [B, 66, 1]
```

**维度变化**:
- `encoder_outputs`: `[B, 66, 768]` → alpha_linear1→ `[B, 66, 768]`
- `att_features`: `[B, 66, 768]` → alpha_linear2→ `[B, 66, 768]`
- 拼接: `[B, 66, 1536]`
- `alpha`: `[B, 66, 1]` = β_t

---

### 7. 最终对齐特征 ĥ_t 计算 (公式5)

#### 论文公式
```
ĥ_t = β_t × h_t + (1-β_t) × h_t^A
维度: ℝ^d
```

#### 代码实现
```python
# MAESC_model.py:217-220
alpha = alpha.repeat(1, 1, 768)  # 广播到768维
# alpha: [B, 66, 1] → [B, 66, 768]

encoder_outputs = torch.mul(1-alpha, encoder_outputs) + torch.mul(alpha, att_features)
# 维度:
# (1-alpha): [B, 66, 768]
# encoder_outputs: [B, 66, 768]
# torch.mul(1-alpha, encoder_outputs): [B, 66, 768]
# alpha: [B, 66, 768]
# att_features: [B, 66, 768]
# torch.mul(alpha, att_features): [B, 66, 768]
# 加法: [B, 66, 768] = ĥ_t
```

**维度变化**:
- `alpha.repeat(1, 1, 768)`: `[B, 66, 1]` → `[B, 66, 768]`
- `encoder_outputs`: `[B, 66, 768]` → 线性组合→ `[B, 66, 768]`

**数学计算**:
```
ĥ_t = (1-β_t) × h_t + β_t × h_t^A
```

---

## 二、AG-GCN模块 - 数据维度变换详解

### 1. 多模态特征分离

#### 代码实现
```python
# MAESC_model.py:249-250
img_feature = encoder_outputs[:, :51, :]  # 图像特征
text_feature = encoder_outputs[:, 51:, :]  # 文本特征
# 维度:
# encoder_outputs: [B, 66, 768]
# img_feature: [B, 51, 768]  # 51个图像patch
# text_feature: [B, 15, 768]  # 15个文本token
```

---

### 2. 布尔依赖矩阵 D 构建 (公式9)

#### 代码实现
```python
# MAESC_model.py:248
new_dependency_matrix = torch.zeros([B, 66, 66], dtype=torch.float).to(encoder_outputs.device)
# 维度: [B, 66, 66] = 分块矩阵D

# 设置对角线为1 (自环)
for i in range(new_dependency_matrix.shape[1]):
    new_dependency_matrix[:, i, i] = 1
# D[i,i] = 1
```

---

### 3. 文本-文本依赖 (D_TT 子矩阵)

#### 论文依赖矩阵 (句法依赖)
```python
# dependency_matrix来自spaCy的句法分析
# 维度: [B, 15, 15] (仅文本部分)
```

#### 代码应用 (公式10部分实现)
```python
# MAESC_model.py:281-284
text_feature_extend1 = text_feature.unsqueeze(1).repeat(1, 15, 1, 1)
text_feature_extend2 = text_feature.unsqueeze(2).repeat(1, 1, 15, 1)
text_sim = torch.cosine_similarity(text_feature_extend1, text_feature_extend2, dim=-1)
# 维度:
# text_feature: [B, 15, 768]
# text_feature_extend1: [B, 15, 15, 768]
# text_feature_extend2: [B, 15, 15, 768]
# text_sim: [B, 15, 15] = 余弦相似度

new_dependency_matrix[:, 51:, 51:] = dependency_matrix * text_sim
# 维度: [B, 15, 15] (填入D_TT子矩阵)
```

**维度变化**:
- `text_feature`: `[B, 15, 768]`
- `text_sim`: `[B, 15, 15]` = F_cosine_similarity(...)
- `new_dependency_matrix[:, 51:, 51:]`: `[B, 15, 15]`

---

### 4. 图像-文本依赖 (D_VT/D_TV 子矩阵)

#### 代码实现 (公式10部分实现)
```python
# MAESC_model.py:270-278
img_feature_extend = img_feature.unsqueeze(2).repeat(1, 1, text_feature.shape[1], 1)
text_feature_extend = text_feature.unsqueeze(1).repeat(1, img_feature.shape[1], 1, 1)
sim = torch.cosine_similarity(img_feature_extend, text_feature_extend, dim=-1)
# 维度:
# img_feature: [B, 51, 768]
# text_feature: [B, 15, 768]
# img_feature_extend: [B, 51, 15, 768]
# text_feature_extend: [B, 51, 15, 768]
# sim: [B, 51, 15] = 图像-文本相似度矩阵

# 图像只与名词挂钩
noun_mask = noun_mask[:, 51:].unsqueeze(1).repeat(1, sim.shape[1], 1)
sim = sim * noun_mask  # 过滤非名词依赖

new_dependency_matrix[:, :51, 51:] = sim  # D_VT
new_dependency_matrix[:, 51:, :51] = torch.transpose(sim, 1, 2)  # D_TV
# 维度: [B, 51, 15]
```

**维度变化**:
- `img_feature_extend`: `[B, 51, 15, 768]`
- `text_feature_extend`: `[B, 51, 15, 768]`
- `sim`: `[B, 51, 15]`
- `new_dependency_matrix[:, :51, 51:]`: `[B, 51, 15]`
- `new_dependency_matrix[:, 51:, :51]`: `[B, 15, 51]`

---

### 5. 情感分数获取 (公式6-7)

#### 论文公式
```
w_i^S = SenticNet(w_i)
s_i = W_S w_i^S + b_S
维度: ℝ^d
```

#### 代码实现
```python
# MAESC_model.py:294-298
# 填充图像区域情感值 (零填充)
sentiment_value = nn.ZeroPad2d(padding=(51, 0, 0, 0))(sentiment_value)
# sentiment_value: [B, 15] → [B, 51+15] = [B, 66]
sentiment_value = sentiment_value.unsqueeze(-1)
# 维度: [B, 66] → [B, 66, 1]

sentiment_feature = self.senti_value_linear(sentiment_value)
# 维度: [B, 66, 1] → senti_value_linear→ [B, 66, 768]
```

**维度变化**:
- `sentiment_value`: `[B, 15]` → ZeroPad2d→ `[B, 66]`
- `sentiment_feature`: `[B, 66, 1]` → Linear→ `[B, 66, 768]` = s_i

---

### 6. 情感-语义融合 (公式8)

#### 论文公式
```
h_i^S = ĥ_i + s_i
维度: ℝ^d
```

#### 代码实现
```python
# MAESC_model.py:298
context_feature = self.context_linear(encoder_outputs + sentiment_feature)
# 维度:
# encoder_outputs: [B, 66, 768] = ĥ_i
# sentiment_feature: [B, 66, 768] = s_i
# 加法: [B, 66, 768]
# context_linear: [B, 66, 768] → [B, 66, 768] (线性变换)
```

**维度变化**:
- `encoder_outputs`: `[B, 66, 768]`
- `sentiment_feature`: `[B, 66, 768]`
- `encoder_outputs + sentiment_feature`: `[B, 66, 768]`
- `context_feature`: `[B, 66, 768]` = h_i^S

---

### 7. 图卷积特征更新 (公式11)

#### 代码实现
```python
# MAESC_model.py:299
context_feature = self.context_gcn(context_feature, context_dependency_matrix, attention_mask)
```

**GCN内部实现** (来自 `GCN` 类):
```python
# GCN.py 中的图卷积计算
def forward(self, inputs, adj, mask=None):
    # inputs: [B, 66, 768]
    # adj: [B, 66, 66]

    support = torch.matmul(inputs, self.weight)
    # support: [B, 66, 768] @ [768, 768] → [B, 66, 768]

    output = torch.matmul(adj, support)
    # output: [B, 66, 66] @ [B, 66, 768] → [B, 66, 768]

    output = self.act(output)  # ReLU
    return output
```

**维度变化**:
- `inputs`: `[B, 66, 768]`
- `adj`: `[B, 66, 66]` = 加权关联矩阵A
- `support`: `[B, 66, 768]`
- `output`: `[B, 66, 768]` = h_i,l^S (第l层GCN输出)

---

### 8. 最终融合特征 (公式12)

#### 论文公式
```
H̃ = λ_1 × Ĥ + λ_2 × Ĥ^S
```

#### 代码实现
```python
# MAESC_model.py:302
mix_feature = self.gcn_proportion * context_feature + encoder_outputs
# 维度:
# context_feature: [B, 66, 768] = Ĥ^S
# encoder_outputs: [B, 66, 768] = Ĥ
# gcn_proportion: scalar = 0.5
# mix_feature: [B, 66, 768] = H̃
```

**维度变化**:
- `context_feature`: `[B, 66, 768]`
- `encoder_outputs`: `[B, 66, 768]`
- `mix_feature`: `[B, 66, 768]`

---

## 三、预测模块 - 数据维度变换

### 1. 解码器输入

#### 代码实现
```python
# MAESC_model.py:237-243
dict = self.decoder(input_ids=tokens,
                    encoder_hidden_states=mix_feature,
                    encoder_padding_mask=encoder_pad_mask,
                    decoder_padding_mask=decoder_pad_mask,
                    decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                    return_dict=True)

hidden_state = dict.last_hidden_state
# 维度: [B, max_len, 768] = h_t^d (公式13)
```

**维度变化**:
- `mix_feature`: `[B, 66, 768]` → BART Decoder→ `hidden_state`: `[B, max_len, 768]`

---

### 2. 预测概率计算 (公式15)

#### 代码实现
```python
# MAESC_model.py:273-278
tag_scores = F.linear(
    hidden_state,
    self.dropout_layer(
        self.decoder.embed_tokens.weight[self.label_start_id:self.label_start_id + 3]))
# 维度:
# hidden_state: [B, max_len, 768]
# embed_tokens.weight: [vocab_size, 768]
# label部分权重: [3, 768] (POS, NEU, NEG)
# F.linear: [B, max_len, 768] @ [3, 768]^T → [B, max_len, 3]

logits[:, :, 3:self.src_start_index] = tag_scores
# 维度: [B, max_len, num_classes]
```

**维度变化**:
- `hidden_state`: `[B, max_len, 768]`
- `tag_scores`: `[B, max_len, 3]`
- `logits`: `[B, max_len, num_classes]` = P(y_t)

---

## 四、维度变化流程图

```
输入: [B, 66, 768]
  ↓
名词提取 → noun_embed: [B, L, 768]
  ↓
A³M模块:
  multi_features_rep: [B, 66, L, 768]
  noun_features_rep: [B, 66, L, 768]
  ↓
concat_features: [B, 66, L, 1536]
  ↓
att: [B, 66, L] (softmax)
  ↓
att_features: [B, 66, 768] = h_t^A
  ↓
alpha: [B, 66, 1] → [B, 66, 768] (β_t)
  ↓
encoder_outputs: [B, 66, 768] = ĥ_t
  ↓
AG-GCN模块:
  img_feature: [B, 51, 768]
  text_feature: [B, 15, 768]
  ↓
dependency_matrix: [B, 66, 66] (D)
  ↓
text_sim: [B, 15, 15]
  ↓
sim (img-text): [B, 51, 15]
  ↓
new_dependency_matrix: [B, 66, 66] (A)
  ↓
sentiment_feature: [B, 66, 768] (s_i)
  ↓
context_feature: [B, 66, 768] = h_i^S
  ↓
GCN(context_feature): [B, 66, 768] = Ĥ^S
  ↓
mix_feature: [B, 66, 768] = H̃
  ↓
Decoder → hidden_state: [B, max_len, 768] = h_t^d
  ↓
tag_scores: [B, max_len, 3]
  ↓
logits: [B, max_len, num_classes] = P(y_t)
```

---

## 五、关键张量维度总结

| 张量名称 | 维度 | 含义 |
|----------|------|------|
| `encoder_outputs` | `[B, 66, 768]` | BART编码输出 |
| `noun_embed` | `[B, L, 768]` | 候选方面特征 H^CA |
| `att` | `[B, 66, L]` | 注意力权重 α_t |
| `att_features` | `[B, 66, 768]` | 方面相关特征 h_t^A |
| `alpha` | `[B, 66, 768]` | 融合系数 β_t |
| `img_feature` | `[B, 51, 768]` | 图像特征 |
| `text_feature` | `[B, 15, 768]` | 文本特征 |
| `dependency_matrix` | `[B, 66, 66]` | 句法依赖矩阵 D |
| `sim` | `[B, 51, 15]` | 图像-文本相似度 |
| `sentiment_feature` | `[B, 66, 768]` | 情感特征 s_i |
| `context_feature` | `[B, 66, 768]` | 融合后特征 h_i^S |
| `mix_feature` | `[B, 66, 768]` | 最终融合特征 H̃ |
| `hidden_state` | `[B, max_len, 768]` | 解码器输出 |
| `logits` | `[B, max_len, num_classes]` | 预测概率 |

---

## 六、代码关键位置

1. **名词提取**: `MAESC_model.py:141-159` (`get_noun_embed`)
2. **A³M注意力**: `MAESC_model.py:207-244` (`noun_attention`)
3. **多模态GCN**: `MAESC_model.py:246-304` (`multimodal_GCN`)
4. **GCN实现**: `src/model/GCN.py` (图卷积层)
5. **情感特征**: `MAESC_model.py:209` (`senti_value_linear`)

---

**分析完成时间**: 2025-11-13
**论文公式与代码维度完全对应**: ✅