import torch
import torch.nn as nn
import torch.nn.functional as F


class word_embedding(nn.Module):
    """词嵌入层 - 带Layer Norm"""
    def __init__(self, vocab_length, embedding_dim):
        super(word_embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_length, embedding_dim, padding_idx=0)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        return self.layer_norm(embedded)


class RNN_model(nn.Module):
    """RNN模型：多层LSTM结构"""
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        super(RNN_model, self).__init__()
        self.vocab_len = vocab_len
        self.word_embedding = word_embedding
        
        # LSTM层 - 使用2层，batch_first=True便于处理数据
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 输出层
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)
    
    def forward(self, x, is_test=False):
        """
        前向传播
        :param x: (batch_size, seq_len)
        :param is_test: 测试模式
        :return: logits (seq_len*batch_size, vocab_len)
        """
        # 词嵌入
        embedded = self.word_embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        output, _ = self.lstm(embedded)  # (batch_size, seq_len, lstm_hidden_dim)
        
        # 展平为二维
        batch_size, seq_len, hidden_dim = output.shape
        output = output.reshape(-1, hidden_dim)  # (seq_len*batch_size, lstm_hidden_dim)
        
        # 全连接层
        logits = self.fc(output)  # (seq_len*batch_size, vocab_len)
        logits = F.log_softmax(logits, dim=1)
        
        return logits

