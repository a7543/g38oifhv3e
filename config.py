from colossalai.amp import AMP_TYPE
from titans.loss.lm_loss import GPTLMLoss
from titans.model.gpt import GPT
from torch.optim import Adam


BATCH_SIZE = 32
#NUM_MICRO_BATCHES = 128 按需增加
SEQ_LEN = 1024
NUM_EPOCHS = 1
HIDDEN_SIZE = 1024
DEPTH = 20 #暂时不能改

TENSOR_PARALLEL = 1
PIPELINE = 1
TENSOR_PARALLEL_MODE = '1d'

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

loss = dict(
    type=GPTLMLoss,
)


def gpt_cust(**kwargs):
    return GPT(vocab_size=53228, max_position_embeddings=SEQ_LEN, hidden_size=HIDDEN_SIZE, depth=DEPTH, num_heads=1, **kwargs)


model = dict(
    type=gpt_cust,
    checkpoint=True,
)

parallel = dict(
    pipeline=PIPELINE,

    tensor=dict(size=TENSOR_PARALLEL, mode=TENSOR_PARALLEL_MODE),
)
