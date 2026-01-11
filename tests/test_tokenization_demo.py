"""
分词过程演示测试
以句子 "I love this phone" 为例，演示完整的分词流程
"""

def demo_basic_tokenization():
    """演示基本分词过程"""
    print("=" * 80)
    print("分词过程演示: 以句子 'I love this phone' 为例")
    print("=" * 80)
    
    # ======================================
    # 步骤1: 输入准备
    # ======================================
    sentence = "I love this phone"
    img_num = 51  # 假设有51个图像特征
    noun_list = ['NNP', 'NNPS', 'NN', 'NNS']
    
    print("\n【步骤1: 输入准备】")
    print(f"  输入句子: '{sentence}'")
    print(f"  图像特征数量: {img_num}")
    print(f"  名词POS标签: {noun_list}")
    
    # ======================================
    # 步骤2: 文本分词
    # ======================================
    sentence_split = sentence.split()
    print(f"\n【步骤2: 文本分词】")
    print(f"  分词结果: {sentence_split}")
    
    # ======================================
    # 步骤3: POS标注 (模拟spaCy结果)
    # ======================================
    # 模拟spaCy的POS标注结果
    pos_results = [
        ('I', 'PRP', False),      # 人称代词，非名词
        ('love', 'VBP', False),   # 动词，非名词
        ('this', 'DT', False),     # 限定词，非名词
        ('phone', 'NN', True)      # 名词，单数
    ]
    
    print(f"\n【步骤3: POS标注】")
    noun_positions = []
    for i, (word, pos, is_noun) in enumerate(pos_results):
        print(f"  '{word}': POS={pos}, 是否名词={is_noun}")
        if is_noun:
            noun_positions.append(i)
    print(f"  名词位置索引: {noun_positions}")
    
    # ======================================
    # 步骤4: BPE分词 (模拟BART分词器)
    # ======================================
    # 模拟BART分词器的BPE分词结果
    bpe_results = {
        'I': [100],
        'love': [200, 201],
        'this': [300],
        'phone': [400, 401]
    }
    
    print(f"\n【步骤4: BPE分词】")
    word_bpes = [50276]  # BOS token ID
    noun_mask = [0]      # BOS不是名词
    
    for j, word in enumerate(sentence_split):
        bpes = bpe_results[word]
        is_noun = j in noun_positions
        
        print(f"  词: '{word}' -> BPE tokens: {bpes}, 是否名词: {is_noun}")
        
        # 名词掩码处理
        if is_noun:
            noun_mask.extend([1] * len(bpes))
            print(f"    -> 名词掩码扩展: {[1] * len(bpes)}")
        else:
            noun_mask.extend([0] * len(bpes))
            print(f"    -> 名词掩码扩展: {[0] * len(bpes)}")
        
        word_bpes.extend(bpes)
    
    word_bpes.append(50277)  # EOS token ID
    noun_mask.append(0)      # EOS不是名词
    
    print(f"  \n最终文本token序列: {word_bpes}")
    print(f"  最终名词掩码: {noun_mask}")
    print(f"  文本序列长度: {len(word_bpes)}")
    
    # ======================================
    # 步骤5: 图像特征编码
    # ======================================
    begin_img = "<<img>>"
    end_img = "<</img>>"
    img_feat = "<<img_feat>>"
    img_feat_id = 50273  # 示例ID
    
    print(f"\n【步骤5: 图像特征编码】")
    image_text = begin_img + img_feat * img_num + end_img
    print(f"  图像token序列: '{image_text[:30]}...{image_text[-10:]}'")
    print(f"  图像特征数量: {img_num}")
    print(f"  图像特征token ID: {img_feat_id}")
    
    # 模拟图像编码结果
    image_ids = [img_feat_id] * img_num
    print(f"  编码后图像ID (前5个): {image_ids[:5]}")
    print(f"  编码后图像ID (后5个): {image_ids[-5:]}")
    
    # ======================================
    # 步骤6: 图像与文本拼接
    # ======================================
    print(f"\n【步骤6: 图像与文本拼接】")
    print(f"  图像序列长度: {len(image_ids)}")
    print(f"  文本序列长度: {len(word_bpes)}")
    
    final_input_ids = image_ids + word_bpes
    print(f"  拼接后总长度: {len(final_input_ids)}")
    print(f"  最终序列 (前10个): {final_input_ids[:10]}")
    print(f"  最终序列 (中间10个): {final_input_ids[50:60]}")
    print(f"  最终序列 (后10个): {final_input_ids[-10:]}")
    
    # ======================================
    # 步骤7: 生成各种掩码
    # ======================================
    print(f"\n【步骤7: 生成掩码】")
    
    # 注意力掩码
    attention_mask = [1] * len(final_input_ids)
    print(f"  注意力掩码 (全1): 长度={len(attention_mask)}")
    
    # 名词掩码
    image_noun_mask = [0] * img_num
    final_noun_mask = image_noun_mask + noun_mask
    print(f"  名词掩码: 图像部分={image_noun_mask}, 文本部分={noun_mask}")
    print(f"  名词掩码长度: {len(final_noun_mask)}")
    print(f"  名词掩码中的1的位置: {[i for i, v in enumerate(final_noun_mask) if v == 1]}")
    
    # 图像掩码
    img_mask = [True if i < img_num else False for i in range(len(final_input_ids))]
    print(f"  图像掩码: True的位置=[0, 1, ..., {img_num-1}], 长度={len(img_mask)}")
    
    # 句子掩码 (仅文本部分)
    sentence_mask = [False] * img_num + [True] * len(word_bpes)
    print(f"  句子掩码: True的位置=[{img_num}, ..., {len(final_input_ids)-1}], 长度={len(sentence_mask)}")
    
    # ======================================
    # 步骤8: 依存关系矩阵 (GCN)
    # ======================================
    print(f"\n【步骤8: 依存关系矩阵 (GCN)】")
    print(f"  矩阵维度: {len(word_bpes)}x{len(word_bpes)} (仅文本部分)")
    print(f"  非零元素示例:")
    print(f"    - 对角线: 5 (自连接，权重最高)")
    print(f"    - love(索引1) -> I(索引0): 1 (主谓关系)")
    print(f"    - love(索引1) -> phone(索引4): 1 (动宾关系)")
    print(f"    - this(索引2) -> phone(索引4): 1 (修饰关系)")
    
    # 显示矩阵结构
    print(f"  \n  依存关系矩阵 (文本部分):")
    for i in range(len(word_bpes)):
        row = []
        for j in range(len(word_bpes)):
            if i == j:
                row.append(5)
            elif (i == 1 and j == 0) or (i == 1 and j == 4) or (i == 2 and j == 4):
                row.append(1)
            else:
                row.append(0)
        print(f"    {row}")
    
    # ======================================
    # 步骤9: SenticNet情感值 (可选)
    # ======================================
    print(f"\n【步骤9: SenticNet情感值 (可选)】")
    print(f"  情感值序列长度: {len(final_input_ids)}")
    print(f"  示例情感值 (前5个): [0.0, 0.0, 0.0, 0.0, 0.0] (所有非名词词为0)")
    
    # ======================================
    # 最终输出
    # ======================================
    print("\n" + "=" * 80)
    print("最终编码结果汇总:")
    print("=" * 80)
    
    result = {
        'input_ids': final_input_ids,
        'attention_mask': attention_mask,
        'noun_mask': final_noun_mask,
        'dependency_matrix': f"({len(word_bpes)}x{len(word_bpes)})",
        'sentiment_value': [0.0] * len(final_input_ids),
        'img_mask': img_mask,
        'sentence_mask': sentence_mask
    }
    
    print(f"\n键值对:")
    for key, value in result.items():
        if isinstance(value, list):
            if len(value) > 10:
                print(f"  {key}: ")
                print(f"    前5个: {value[:5]}")
                print(f"    后5个: {value[-5:]}")
                print(f"    长度: {len(value)}")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("关键统计信息:")
    print("=" * 80)
    print(f"  输入序列总长度: {len(final_input_ids)}")
    print(f"  图像特征数量: {img_num}")
    print(f"  文本token数量: {len(word_bpes)}")
    print(f"  名词数量: {sum(final_noun_mask)}")
    print(f"  BPE展开后的依存矩阵: {len(word_bpes)}x{len(word_bpes)}")
    
    return result


def demo_aesc_label():
    """演示AESC标签编码过程"""
    print("\n" + "=" * 80)
    print("AESC标签编码演示:")
    print("=" * 80)
    
    # 输入：句子和方面-情感标注
    sentence = "I love this phone"
    aesc_spans = [(1, 3, 'POS')]  # (起始位置, 结束位置, 情感极性)
    
    print(f"\n【输入】")
    print(f"  句子: '{sentence}'")
    print(f"  方面-情感标注: {aesc_spans}")
    print(f"    解析: 方面='{' '.join(sentence.split()[1:3])}' (love), 情感='POS'")
    
    # AESC编码格式
    print(f"\n【AESC编码格式】")
    print(f"  格式: [<<text>>] [text words] [<</text>>] [AESC] [start] [end] [polarity] [EOS]")
    
    # 模拟编码过程
    print(f"\n【编码步骤】")
    print(f"  1. 添加文本标记: <<text>>")
    print(f"  2. 添加文本token: I, love, this, phone")
    print(f"  3. 添加文本结束: <</text>>")
    print(f"  4. 添加AESC标记")
    print(f"  5. 添加方面span: start=?, end=?, polarity=?")
    
    # 计算span位置
    word_positions = [0, 1, 2, 3]  # 4个词
    print(f"\n  词位置映射: {dict(zip(sentence.split(), word_positions))}")
    print(f"  方面 'love' 位置: 起始={1}, 结束={3} (不包括3)")
    print(f"  情感极性: POS -> ID=3")
    
    # 最终AESC输出
    print(f"\n【AESC最终输出】")
    aesc_output = [
        "<<text>>",      # 文本开始标记
        "I", "love", "this", "phone",  # 文本内容
        "<</text>>",     # 文本结束标记
        "<<AESC>>",      # AESC任务标记
        "1002",          # 方面起始位置 (示例)
        "1005",          # 方面结束位置 (示例)
        "3",             # 情感极性ID (POS)
        "1"              # EOS标记
    ]
    
    print(f"  Token序列: {aesc_output}")
    print(f"  序列长度: {len(aesc_output)}")
    
    # 转换为ID序列
    id_mapping = {
        "<<text>>": 1000,
        "I": 100,
        "love": 200,
        "this": 300,
        "phone": 400,
        "<</text>>": 1001,
        "<<AESC>>": 5000,
        "1": 50277  # EOS
    }
    
    print(f"\n  ID序列: {[id_mapping.get(tok, tok) for tok in aesc_output]}")


if __name__ == "__main__":
    # 运行基本分词演示
    result = demo_basic_tokenization()
    
    # 运行AESC标签演示
    demo_aesc_label()
    
    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)
