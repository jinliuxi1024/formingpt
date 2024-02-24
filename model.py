from __future__ import annotations
# annotations 是用于做函数注解的
import math
import torch
import torch.nn as nn


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5*x*(1.0+torch.tanh(math.sqrt(2.0/math.pi)*(x+0.044715*torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    # 前序自注意力机制
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_length: int,
        dropout_prob: float=0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_length = max_length
        self.dropout_prob = dropout_prob
        # 这个是神经元的丢弃概率
        self.input_projection = nn.Linear(in_features=hidden_size,out_features=3*hidden_size)
        self.output_projection = nn.Linear(in_features=hidden_size,out_features=hidden_size)
        # 这里是横向拓展 2x3的矩阵变化为 6x3的矩阵
        self.attention_dropout = nn.Dropout(p=dropout_prob)
        self.output_dropout = nn.Dropout(p=dropout_prob)
        # 这个是自动的
        self.causal_mask = torch.tril(torch.ones(max_length,max_length)).view(1,1,max_length,max_length)
        # 在这里只要下三角矩阵，减少对未来的注意力

    def forward(self, input_tensor: torch.Tensor):
        # input_tensor 的结构是[batch_size,sequence_length,hidden_size]
        batch_size, sequence_length, hidden_size = input_tensor.size()
        # batch_size 为 批次样本的数量，sequence_length为样本的长度
        # projected tensor的结构为[batch_size,sequence_length,3*hidden_size]
        projected_tensor = self.input_projection(input_tensor)
        query, key, value = torch.chunk(projected_tensor, chunks=3,dim=-1)
        # 这个是切块函数
        # 从总体来看，将原来的输入矩阵拓展为三份后再变为不同的矩阵
        hidden_size_per_head = hidden_size//self.num_heads
        query = query.view(batch_size, sequence_length, self.num_heads, hidden_size_per_head).transpose(1, 2)
        # query的结构转换成[batch_size, num_heads, sequence_length, hidden_size_per_head]
        key = key.view(batch_size, sequence_length, self.num_heads, hidden_size_per_head).transpose(1, 2).transpose(-2, -1)
        # key的结构转换为[batch_size, num_heads, hidden_size_per_head, sequence_length]
        value = value.view(batch_size, sequence_length, self.num_heads, hidden_size_per_head).transpose(1,2)
        # value的结构转换为[batch_size, num_heads, sequence_length, hidden_size_per_head]
        attention_scores =torch.matmul(input=query, other=key)/math.sqrt(hidden_size_per_head)
        # 在这里attention_scores的结构为[batch_size, num_heads, sequence_length, sequence_length]
        masked_attention_scores = attention_scores.masked_fill(
            self.causal_mask[:,:,:sequence_length,:sequence_length] == 0, float('-inf')
        )
        # 在这里应用掩码对注意力矩阵对未来的位置掩盖
        attention_prob = torch.nn.functional.softmax(masked_attention_scores,dim=-1)
        attention_prob = self.attention_dropout(attention_prob)
        output = torch.matmul(attention_prob, value)
        # 这里output的结构为[batch_size, num_heads, sequence_length, hidden_size_per_head]
        # 这里是将注意力概率与value相乘
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_size)
        # 这里转换的下标是从0开始的
        # output的结构为[batch_size, sequence_length, hidden_size]
        output = self.output_dropout(self.output_projection(output))
        return output


class  FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout_prob: float=0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(in_features=hidden_size, out_features=4 * hidden_size)
        self.activation = NewGELU()
        self.output_projection = nn.Linear(in_features=4 * hidden_size, out_features=hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input_tensor: torch.Tensor):
        # input_tensor的结构为[batch_size, sequence_length, hidden_size]
        output = self.output_projection(self.activation(self.input_projection(input_tensor)))
        output = self.dropout(output)
        return output


class GPTBlock(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            max_length: int,
            dropout_prob: float = 0.1,
            # 总体来说，这是初始化的参数
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_length = max_length
        self.dropout_prob = dropout_prob

        self.input_layer_norm = nn.LayerNorm(hidden_size)
        # 这个是输入层的归一化
        self.intermediate_layer_norm = nn.LayerNorm(hidden_size)
        # 这个是中间层的归一化
        # 可以这么认为都是构建对象来实现操作
        self.attention = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_length=max_length,
            dropout_prob=dropout_prob,
        )

        self.feedforward = FeedForward(hidden_size=hidden_size, dropout_prob=dropout_prob)

    def forward(self,input_tensor: torch.Tensor):
        # input_tensor的结构为[batch_size, sequence_length, hidden_size]
        # output的结构为[batch_size, sequence_length, hidden_size]
        output = input_tensor + self.attention(self.input_layer_norm(input_tensor))
        # 这里是输入层的归一化
        output = output + self.feedforward(self.intermediate_layer_norm(output))
        # 这里是中间层的归一化
        return output


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        max_length: int,
        num_layers: int,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        # 这里是词嵌入
        self.position_embedding = nn.Embedding(num_embeddings=max_length, embedding_dim=hidden_size)
        # 这里是位置编码
        self.dropout = nn.Dropout(p=dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList(
            [
                GPTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    max_length=max_length,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_layers)
            ]
        )
        # 这里是多层的GPTBlock

    def forward(self,input_ids: torch.Tensor):
        device = input_ids.device
        # 这个是设备
        sequence_length = input_ids.size(1)

        position_ids = torch.arange(sequence_length, dtype=torch.long, device=device).unsqueeze(0)
        # arange是生成一个序列,unsequeeze是增加一个维度
        token_embeddings = self.token_embedding(input_ids)
        # 这里是词嵌入
        position_embeddings = self.position_embedding(position_ids)
        # 这里是位置编码

        hidden_states = self.dropout(token_embeddings + position_embeddings)
        # 这里是词嵌入和位置编码的相加
        for block in self.blocks:
            hidden_states = block(hidden_states)
            # 这里是多层的GPTBlock
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


# 这里是GPT模型的定义
class GPTForLanguageModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            num_heads: int,
            max_length: int,
            num_layers: int,
            # num_layers是层数
            dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.transformer = GPTModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_length=max_length,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
        )
        # 这是gpt转换器
        self.language_model_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # 无偏置的线性层，这里是语言模型的头部
        self.criterion = nn.CrossEntropyLoss()
        # 交叉熵损失函数,用于计算损失,交叉熵损失函数是用于多分类问题的损失函数，它的计算公式是y=-log(p)，p是模型输出的概率值，y是真实标签的概率值
        self.apply(self._init_weights)
        # 这是初始化权重

    def _init_weights(self, module):
        # 这个初始化权重的函数可以根据传入参数的类型来初始化权重
        if isinstance(module, nn.Linear):
        # 如果是线性层,就初始化权重
           torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
           # normal_是正态分布的初始化
           if module.bias is not None:
               torch.nn.init.zeros_(module.bias)
                # zeros_是零初始化
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 如果是词嵌入,就初始化权重
            if module.padding_idx is not None:
            # padding_idx是填充索引
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            # 如果是归一化,就初始化权重
            module.weight.data.fill_(1.0)
            # fill_是填充初始化

        for name, parameter in module.named_parameters():
            if name == 'output_projection.weight':
            # 如果是输出投影的权重,就初始化权重
                torch.nn.init.normal_(parameter, mean=0.0, std=(0.02/math.sqrt(2 * self.num_layers)))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        hidden_states = self.transformer(input_ids)
        # input_ids,labels的结构为[batch_size, sequence_length]
        # hidden_states的结构为[batch_size, sequence_length, hidden_size]
        logits = self.language_model_head(hidden_states)
        # logits为初始的预测值
        output = {
            'logits': logits,
        }
        # 这里是输出的字典
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
        # 这行代码的目的是对 logits 进行切片操作，选择除了倒数第二个维度的最后一个元素之外的所有元素，并确保结果张量是连续存储的
        # 这里是对logits进行切片操作，shift_logits是切片后的logits
            shift_labels = labels[..., 1:].contiguous()
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # 这里是计算损失,view(-1)是将张量展平成一维张量
            output['loss'] = loss
        return output

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> GPTForLanguageModel:
        from typing import cast
        # cast函数是用来指明参数类型的
        from transformers import GPT2LMHeadModel
        from mymingpt.utils import adapt_huggingface_transformers_state_dict

        huggingface_model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        huggingface_model = cast(GPT2LMHeadModel, huggingface_model)
        # cast函数是用来指明参数类型的
        model = GPTForLanguageModel(
            vocab_size=huggingface_model.config.vocab_size,
            hidden_size=huggingface_model.config.n_embd,
            num_heads=huggingface_model.config.n_head,
            max_length=huggingface_model.config.n_ctx,
            num_layers=huggingface_model.config.n_layer,
            dropout_prob=huggingface_model.config.attn_pdrop,
        )
        state_dict = huggingface_model.state_dict()
        state_dict = adapt_huggingface_transformers_state_dict(state_dict)
        model.load_state_dict(state_dict)
        model = model.eval()
        # 这里是加载预训练模型，切换到推断模式
        return model

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, num_new_tokens: int =64, temperature: float = 1.0):
        for _ in range(num_new_tokens):
            logits = self(prompt_ids)['logits']
            logits = logits[:,-1, :]/temperature
            probs = torch.softmax(logits,dim=-1)
            new_token_id = torch.multinomial(probs, num_samples=1)
            # 采样一个新的token
            prompt_ids = torch.cat((prompt_ids, new_token_id), dim=1)
        return prompt_ids












