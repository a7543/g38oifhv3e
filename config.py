from colossalai.amp import AMP_TYPE
from titans.loss.lm_loss import GPTLMLoss
from titans.model.gpt import GPT
from torch.optim import Adam

TENSOR_PARALLEL = 1

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

loss = dict(
    type=GPTLMLoss,
)

parallel = dict(
    pipeline=1,
    tensor=dict(size=TENSOR_PARALLEL, mode=None),
)
