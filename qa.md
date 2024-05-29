### backbone输出两个值，一个feature，一个pos emb，为什么pos emb要在backbone里做
DETR backbone: Joiner --> BackboneBase & PositionEmbeddingSine
mask参与了pos embed的计算，对于pad部分忽略，pos embed对应元素为0
### 为什么feature有两个
两个分别是tensors（特征图）和mask
### pos embed参与了transformer里面的哪些计算？
encoder和decoder都会输入
### encoder的pos embed作用

### decoder的pos和query pos的区别