from colossalai.amp import AMP_TYPE
from titans.loss.lm_loss import GPTLMLoss
from titans.model.gpt import GPT
from torch.optim import Adam


BATCH_SIZE = 1
SEQ_LEN = 2048
NUM_EPOCHS = 1

TENSOR_PARALLEL = 2
PIPELINE = 2

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

model = dict(
    type=GPT(dim=3072, depth=40, num_heads=24),
    checkpoint=True,
)

parallel = dict(
    pipeline=PIPELINE,
    tensor=dict(size=TENSOR_PARALLEL, mode='1d'),
)
