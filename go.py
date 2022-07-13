import os

import colossalai
import colossalai.utils as utils
from colossalai.nn import LinearWarmupLR
from colossalai.trainer import Trainer, hooks
from titans.loss.lm_loss import GPTLMLoss

from titans.model.gpt import GPT
from torch.optim import Adam
from yuan import YuanDataset


def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()

    colossalai.launch_from_torch(config=args.config)

    print('Build data loader')

    train_ds = YuanDataset(
        os.environ['DATA'], vocab_path='/home/nfs/zsc/yuan/coldata_dbg/vocab.txt', seq_len=1262)
    train_dataloader = utils.get_dataloader(
        train_ds, seed=42, batch_size=32, shuffle=True)
        
    print('Build model')

    model = GPT(vocab_size=53228, hidden_size=16,
                max_position_embeddings=1262, depth=2, num_heads=4)

    criterion = GPTLMLoss()

    print('Build optimizer')
    optimizer = Adam(model.parameters(), lr=0.00015, weight_decay=1e-2)

    lr_scheduler = LinearWarmupLR(optimizer, total_steps=10, warmup_steps=5)

    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
                                                                      optimizer,
                                                                      criterion,
                                                                      train_dataloader=train_dataloader,
                                                                      lr_scheduler=lr_scheduler)

    trainer = Trainer(engine=engine)

    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=10,
                test_interval=1,
                hooks=hook_list,
                display_progress=True,
                return_output_label=False)


if __name__ == '__main__':
    main()
