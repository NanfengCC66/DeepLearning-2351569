import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pypinyin import pinyin, Style
import rnn as rnn_lstm

start_token = 'G'
end_token = 'E'


def process_poems_seven_only(file_name, max_poems=None):
    """
    专门提取标准七言绝句 (4句, 每句7字, 总长32)
    """
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                line_content = line.strip().split(':')
                if len(line_content) != 2:
                    continue
                title, content = line_content
                content = content.replace(' ', '')
                
                # 过滤非法字符
                if any(c in content for c in ['_', '(', '（', '《', '[', start_token, end_token]):
                    continue
                
                # 精准匹配：七言绝句通常包含 4 句，每句 7 字 + 1 个标点 = 8 字符
                # 总长度应该是 8 * 4 = 32 字符
                if len(content) == 32 and content[7] in '，。' and content[15] in '，。':
                    content = start_token + content + end_token
                    poems.append(content)
                
                if max_poems and len(poems) >= max_poems:
                    break
            except (ValueError, IndexError):
                pass
    
    # 统计信息
    print(f"  - 匹配到的七言绝句数量: {len(poems)}")
    
    # 后续的词汇表构建逻辑保持不变
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def get_rhyme_group(char):
    """获取汉字的韵母组
    :param char: 单个汉字
    :return: 韵母字符串（如 'an', 'ang', 'ong' 等）
    """
    try:
        res = pinyin(char, style=Style.FINALS, strict=False)
        return res[0][0] if res else None
    except:
        return None


class PoemDataset(Dataset):
    """诗歌数据集"""
    def __init__(self, poems_vector):
        self.poems = poems_vector
    
    def __len__(self):
        return len(self.poems)
    
    def __getitem__(self, idx):
        poem = self.poems[idx]
        x = poem[:-1]
        y = poem[1:]
        return torch.LongTensor(x), torch.LongTensor(y)


def collate_fn(batch):
    """填充批次数据到相同长度"""
    x_list, y_list = zip(*batch)
    max_len = max(len(x) for x in x_list)
    
    x_padded = []
    y_padded = []
    for x, y in zip(x_list, y_list):
        pad_len = max_len - len(x)
        x_padded.append(torch.cat([x, torch.zeros(pad_len, dtype=torch.long)]))
        y_padded.append(torch.cat([y, torch.zeros(pad_len, dtype=torch.long)]))
    
    return torch.stack(x_padded), torch.stack(y_padded)


def run_training():
    # 加载七言绝句数据 - 使用更多样本让模型学习自然的节奏
    poems_vector, word_to_int, vocabularies = process_poems_seven_only('./poems.txt', max_poems=None)
    print(f"✓ 加载七言绝句数据完成")
    print(f"  - 诗歌数: {len(poems_vector)}")
    print(f"  - 词汇量: {len(word_to_int)}")
    
    BATCH_SIZE = 32
    dataset = PoemDataset(poems_vector)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           collate_fn=collate_fn, num_workers=0)
    
    torch.manual_seed(5)
    
    vocab_size = len(word_to_int) + 1
    word_embedding = rnn_lstm.word_embedding(vocab_length=vocab_size, embedding_dim=128)
    rnn_model = rnn_lstm.RNN_model(batch_sz=BATCH_SIZE, vocab_len=vocab_size,
                                   word_embedding=word_embedding, embedding_dim=128, 
                                   lstm_hidden_dim=256)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rnn_model = rnn_model.to(device)
    print(f"✓ 使用设备: {device}\n")
    
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    loss_fun = nn.NLLLoss(ignore_index=0)
    
    # 训练参数 - 更激进的策略用于高质量数据
    num_epochs = 100
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 8
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (x_tensor, y_tensor) in enumerate(dataloader):
            x_tensor = x_tensor.to(device)
            y_tensor = y_tensor.to(device)
            
            # 前向传播
            logits = rnn_model(x_tensor)
            loss = loss_fun(logits, y_tensor.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # 定期输出
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1:3d} | Batch {batch_idx:2d}/{len(dataloader):2d} | Loss: {loss.item():.4f}")
            
            if batch_idx == 0:
                _, pred = torch.max(logits.view(-1, vocab_size), dim=1)
                target_sample = y_tensor[0][:5].tolist()
                print(f"  └─ Pred: {pred[:5].tolist()} | True: {target_sample}")
        
        avg_loss = epoch_loss / batch_count
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1:3d} 完成 | Avg Loss: {avg_loss:.4f}")
        print(f"{'='*60}\n")
        
        # 学习率调度
        scheduler.step(avg_loss)
        
        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(rnn_model.state_dict(), './poem_generator_rnn')
            print(f"✓ 保存模型 (Loss: {best_loss:.4f})\n")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"✗ Loss 连续 {patience_limit} 个 Epoch 未改善，提前停止训练")
                break
        
        # 提前停止条件 - 降低目标，因为是高质量数据
        if avg_loss < 2.5:
            print(f"✓ Loss 已降至 {avg_loss:.4f}，达到目标（高质量数据），停止训练")
            break
    
    print(f"\n✓ 训练完成！最佳 Loss: {best_loss:.4f}")


def sample_with_repetition_penalty(logits, generated_ids, penalty=1.2, temperature=1.0, top_k=50):
    """带重复惩罚的采样 - 防止词汇重复
    :param logits: 原始logits (vocab_size,)
    :param generated_ids: 已经生成的字符 ID 列表
    :param penalty: 惩罚系数，>1 表示降低已出现词的概率
    :param temperature: 温度系数，<1时更确定，>1时更随机
    :param top_k: 只从top-k个最高概率中采样
    :return: 采样出的词索引
    """
    logits = logits.clone()
    
    # 对已经出现过的字符降低其 logits 值
    for char_id in set(generated_ids):
        if char_id < len(logits):
            if logits[char_id] > 0:
                logits[char_id] /= penalty
            else:
                logits[char_id] *= penalty
    
    logits = logits / temperature
    
    # Top-K 过滤
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float('Inf')
    
    probs = F.softmax(logits, dim=-1)
    probs = torch.clamp(probs, min=1e-10)
    return torch.multinomial(probs, 1).item()


def sample_with_rhyme(logits, target_rhyme=None, vocabularies=None, temperature=1.0, penalty=1e9):
    """带韵脚约束的采样 - 强制押韵
    :param logits: 原始logits
    :param target_rhyme: 目标韵母（如 'an', 'ang'）
    :param vocabularies: 词汇表
    :param temperature: 温度系数
    :param penalty: 对不押韵字的惩罚系数
    :return: 采样出的词索引
    """
    logits = logits.clone()
    
    if target_rhyme and vocabularies:
        # 遍历词表，把不属于该韵部的字概率降为极低
        for i, word in enumerate(vocabularies):
            if i == 0:  # 跳过 padding
                continue
            word_rhyme = get_rhyme_group(word)
            if word_rhyme != target_rhyme:
                logits[i] -= penalty  # 惩罚不押韵的字
    
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    probs = torch.clamp(probs, min=1e-10)
    return torch.multinomial(probs, 1).item()


def to_word(predict, vocabs):
    """预测向量转化为汉字"""
    if isinstance(predict, torch.Tensor):
        sample = torch.argmax(predict).item()
    else:
        sample = np.argmax(predict)
    
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def pretty_print_poem(poem):
    """打印诗歌，去掉特殊标记"""
    shige = []
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    return ''.join(shige)


def gen_poem_final(begin_word, temperature=0.8):
    """最终版生成函数 - 整合自动标点、韵脚记忆和重复惩罚
    :param begin_word: 开头字
    :param temperature: 采样温度
    """
    poems_vector, word_int_map, vocabularies = process_poems_seven_only('./poems.txt', max_poems=None)
    vocab_size = len(word_int_map) + 1
    
    word_embedding = rnn_lstm.word_embedding(vocab_length=vocab_size, embedding_dim=128)
    rnn_model = rnn_lstm.RNN_model(batch_sz=1, vocab_len=vocab_size,
                                   word_embedding=word_embedding, embedding_dim=128, 
                                   lstm_hidden_dim=256)
    
    rnn_model.load_state_dict(torch.load('./poem_generator_rnn', map_location='cpu'))
    rnn_model.eval()
    
    device = torch.device('cpu')
    rnn_model = rnn_model.to(device)
    
    # 基础设置
    poem = begin_word
    input_ids = [word_int_map.get(start_token, 0), word_int_map.get(begin_word, 0)]
    generated_ids = [word_int_map.get(begin_word, 0)]
    
    # 标点符号 ID
    comma_id = word_int_map.get('，', 2)
    period_id = word_int_map.get('。', 3)
    
    word_count = 1  # 计数当前句子的字数
    sentence_count = 0
    first_rhyme = None  # 用于存储第二句末尾的韵脚
    
    with torch.no_grad():
        for _ in range(40):
            x = torch.LongTensor([input_ids]).to(device)
            logits = rnn_model(x, is_test=True)[-1]
            
            # --- 核心硬干预逻辑 ---
            if word_count == 7:
                # 满 7 字，强行输出标点
                word_id = comma_id if sentence_count % 2 == 0 else period_id
                
                # 如果是第二句末尾（sentence_count为1时），记录韵脚
                if sentence_count == 1:
                    last_char = vocabularies[generated_ids[-1]]
                    first_rhyme = get_rhyme_group(last_char)
                
                sentence_count += 1
                word_count = 0
            else:
                # 未满 7 字，屏蔽标点和结束符
                logits[comma_id] = -1e9
                logits[period_id] = -1e9
                logits[word_int_map.get(end_token, 0)] = -1e9
                
                # 如果是第四句末尾，尝试匹配第二句韵脚
                current_target_rhyme = first_rhyme if (sentence_count == 3 and word_count == 6) else None
                
                if current_target_rhyme:
                    word_id = sample_with_rhyme(logits, current_target_rhyme, vocabularies, temperature)
                else:
                    word_id = sample_with_repetition_penalty(logits, generated_ids, penalty=1.5, temperature=temperature)
                
                word_count += 1
            # ---------------------
            
            word = vocabularies[word_id]
            poem += word
            input_ids.append(word_id)
            generated_ids.append(word_id)
            
            if sentence_count >= 4:
                break  # 生成四句停止
    
    return poem


if __name__ == '__main__':
    print("\n" + "="*60)
    print("唐诗生成器 - 训练阶段")
    print("="*60 + "\n")
    
    run_training()
    
    print("\n" + "="*60)
    print("唐诗生成器 - 生成阶段（最终版）")
    print("="*60 + "\n")
    
    # 生成多首七言诗，展示效果
    for begin_char in ["日", "红", "山", "夜", "湖", "海", "月"]:
        try:
            poem = gen_poem_final(begin_char, temperature=0.8)
            result = pretty_print_poem(poem)
            print(f"【{begin_char}】{result}")
        except Exception as e:
            print(f"【{begin_char}】生成失败: {str(e)}")
    
    print("\n" + "="*60)
    print("多样性演示 - 同一开头生成多首诗")
    print("="*60 + "\n")
    
    for i in range(3):
        try:
            poem = gen_poem_final("春", temperature=0.9)
            result = pretty_print_poem(poem)
            print(f"【春】({i+1}) {result}")
        except Exception as e:
            print(f"【春】({i+1}) 生成失败: {str(e)}")
    
    print()


