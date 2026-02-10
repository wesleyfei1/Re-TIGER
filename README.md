Re-TIGER 复现计划

目标：从原始数据到可复现的训练、推理与评估流程。所有操作在本仓库下执行，数据集位于 ~/datasets。

DONE: 0. 环境与目录准备
- 依赖管理：conda 环境 tiger
- 已安装依赖：torch、torchvision、torchaudio
- 目录结构已创建：data/、outputs/、checkpoints/、logs/
- Python 版本：3.11
- 随机种子：11
- 硬件：A100-SXM4-80GB
- 系统：Linux
- CUDA：13.0

DONE: 1. 数据准备与预处理 (Data Preprocessing)
Amazon Product Reviews 数据（Beauty）包括Beauty的review以及meta+Beauty(已经存放在~/datasets中)
- 清洗交互数据：Leave-one-out，过滤交互少于 5 次的用户
- 构建物品文本：Title + Price + Brand + Category 拼接
- 划分数据集：train/valid/test，并保存为可复用格式（jsonl）
- 处理脚本：utils/data_process.py
- 输出目录：~/datasets/processed/beauty/
- 本次运行统计：num_users=1620, num_items=6982, num_train=11744, num_valid=1620, num_test=1620

DONE: 2. 语义向量生成 (Semantic Embeddings)
- 模型：Sentence-T5(预训练)
- 输入：物品文本；输出：768 维向量
- 产物（位于 ~/datasets/processed/beauty/）
	- item_text.jsonl: {"item_id": str, "text": str}
	- sentence_t5_embeddings.npy: shape = [num_items, 768], dtype=float32
	- sentence_t5_item_ids.json: [item_id_0, item_id_1, ...] 与 embeddings 按索引一一对应
	- train.jsonl / valid.jsonl / test.jsonl: {"user_id": str, "item_id": str, "timestamp": int}
	- stats.json: 统计信息（num_users/num_items/num_train/num_valid/num_test 等）
- RQ-VAE 使用格式
	- 读取 sentence_t5_embeddings.npy 作为输入特征矩阵 X (N x 768)
	- 读取 sentence_t5_item_ids.json 作为索引映射，保证 embedding[i] 对应 item_id[i]
- TIGER 训练需要的格式
	- 训练/验证/测试序列来自 train.jsonl / valid.jsonl / test.jsonl
	- 使用 RQ-VAE 输出的 ItemID -> SemanticID(4-tuple) 映射表，将序列中的 item_id 转成语义 token 序列

DONE: 3. 语义 ID 生成 (Semantic ID with RQ-VAE)
- 模型结构
	- Encoder: 768 -> 512 -> 256 -> 128 -> 32
	- Residual Quantizer: m=3 层，K=256 codebook
	- Decoder: 32 -> 128 -> 256 -> 512 -> 768
- 训练设置
	- Loss: MSE + Commitment loss
	- 初始化: K-means 预热 codebook 防止坍缩
	- Batch size: 1024, Optimizer: Adagrad, LR: 0.4
- 生成语义 ID
	- 输出 3 位 tuple (c1, c2, c3)
	- 冲突处理：追加第 4 位 token，冲突项标注为 (c1, c2, c3, 0/1/2/...)
- 构建双向映射：ItemID <-> SemanticID(4-tuple)
- 在~/Re-TIGER/init中写codebook的初始化，然后在~/Re-TIGER/models中写rq-vae的代码，然后在~/Re-TIGER/train中书写训练rq-vae的代码
- 训练与使用流程
	- 初始化 codebook（可选但推荐）
		- python init/codebook_init.py --embeddings ~/datasets/processed/beauty/sentence_t5_embeddings.npy --output ~/datasets/processed/beauty/rqvae_codebooks.pt --latent_dim 32
	- 训练 RQ-VAE
		- python train/train_rqvae.py --embeddings ~/datasets/processed/beauty/sentence_t5_embeddings.npy --item_ids ~/datasets/processed/beauty/sentence_t5_item_ids.json --output_dir ~/datasets/processed/beauty --codebook_init_path ~/datasets/processed/beauty/rqvae_codebooks.pt
	- 产物
		- rqvae.pt: 模型权重
		- rqvae_codebooks.pt: codebook 初始化权重
		- semantic_ids.json: ItemID -> SemanticID(3/4-tuple)
		- semantic_id_to_item.json: SemanticID -> ItemID 列表
	- 运行结果（已生成）
		- ~/datasets/processed/beauty/rqvae.pt
		- ~/datasets/processed/beauty/rqvae_codebooks.pt
		- ~/datasets/processed/beauty/semantic_ids.json
		- ~/datasets/processed/beauty/semantic_id_to_item.json

TODO: 4. 生成式检索模型 (Generative Retrieval Model)
- 数据构建
	- 输入序列：[UserIDToken] + 历史 Item 的 SemanticID tokens
	- UserID hashing 到 <= 2000 token
	- 目标：下一个物品的 4-token SemanticID
- 使用 TODO3 产物
	- semantic_ids.json: 将 train/valid/test 中的 item_id 映射为 3/4 token 序列
	- semantic_id_to_item.json: 解码约束与反查 ItemID
- 模型结构（从零训练）
	- 使用Hashing Trick将2000个user id对应到userid token
	- Seq2Seq Transformer 或 HF T5Config
	- Encoder/Decoder 各 4 层，6 heads，hidden dim 64,relu
	- 词表：codebook tokens 1024 + user tokens 2000 + BOS/EOS/PAD
	- dropout=0.1,batch=1024,总共训练200k steps
- 训练
	- 目标：Next token prediction (cross-entropy)
	- 学习率策略：inverse square root decay，前10k steps的时候learning_rate=0.04

TODO: 5. 推理与评估 (Inference & Evaluation)
- Beam search 生成 SemanticID
- 约束解码：过滤无效 tuple（不在映射表中）
- 使用 TODO3 产物
	- semantic_id_to_item.json: 约束解码与候选 ItemID 反查
- 评估指标：Recall@5/10, NDCG@5/10
- 可选：Cold-start 测试（训练中移除部分物品，RQ-VAE 保留）

TODO: 6. 复现结果整理与记录
- 记录每阶段的参数、训练日志与模型检查点
- 汇总实验结果与对比表（不同数据集与设置）
- 输出可复现脚本与运行说明