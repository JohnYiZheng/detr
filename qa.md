### backbone输出两个值，一个feature，一个pos emb，为什么pos emb要在backbone里做
DETR backbone: Joiner --> BackboneBase & PositionEmbeddingSine
mask参与了pos embed的计算，对于pad部分忽略，pos embed对应元素为0
### 为什么feature有两个
两个分别是tensors（特征图）和mask
### pos embed参与了transformer里面的哪些计算？
encoder和decoder都会输入
### encoder的pos embed作用
对特征图特征赋值位置信息

### decoder的pos和query pos的区别
两种不同作用的位置编码
query_pos包含了特定查询的**语义信息**

### transformer中的tgt是干什么用的？
真正的查询序列，组合query_pos后，作为解码器的实际序列

### encoder中mask和src_key_padding_mask有什么区别
先回顾一下naive transformer和vit中encoder做了什么
- naive transformer：
  - mask传导顺序：encoder -- encoder layer -- multiheadattn -- scale prod atten。最终最用在计算atten分数时，在softmax之前，将mask == 0的部分的score设置成负无穷，这样经过softmax后该位置分数为0；
  - mask构造：make_src_mask，将batch等长处理的pad无效数据遮盖掉；
- vit：没有mask

detr的mask是什么？
data_loader_train = DataLoader(... collate_fn=utils.collate_fn ...)
nested_tensor_from_tensor_list中，将有效像素设为false，pad设为true
backbone中，通过F.interpolate将mask差值为和特征图相同的hw
mask走向：detr入参sample输入--self.backbone生成下采样mask--transformer输入--作为encoder的src_key_padding_mask输入 & 作为decoder的memory_key_padding_mask输入

encoder：作为encoder的src_key_padding_mask输入，送入layer,最终作为入参key_padding_mask送入nn.torch.nn.MultiHeadAttention中，forward入参的key_padding_mask和atten_mask有什么区别？

在 `torch.nn.MultiheadAttention` 中，`key_padding_mask` 和 `attn_mask` 这两个参数都可以影响到注意力得分（attention scores）的计算，但它们的作用方式略有不同。

1. `key_padding_mask`:
   - `key_padding_mask` 主要用于处理输入序列中的填充部分，确保在计算注意力得分时不考虑这些填充部分。这意味着，对于填充部分，对应的注意力得分会被设为负无穷大（或者很小的值），从而在 softmax 操作后得到接近于零的注意力权重。
   - `key_padding_mask` 主要影响到了 softmax 操作前的注意力得分计算过程，在这个过程中确保了填充部分不会对注意力权重产生影响。

2. `attn_mask`:
   - `attn_mask` 则是在注意力得分计算之后，通过在注意力权重矩阵中引入屏蔽或限制，从而调整注意力权重的分布。比如，在文本生成任务中，可以使用 `attn_mask` 来屏蔽当前位置之后的位置，这样模型就不会窥探未来信息。
   - `attn_mask` 主要影响到了 softmax 操作后的注意力权重，通过在这一步引入屏蔽或限制，调整了最终的注意力权重分布。

因此，虽然这两个参数都可以影响到注意力得分的计算，但作用的阶段和方式略有不同：`key_padding_mask` 主要在 softmax 操作前处理填充部分，而 `attn_mask` 则在 softmax 操作后对注意力权重进行调整。

在transformer仓中，这两个mask都是给score赋值为负无穷，gpt答案存疑