import time
import torch
import wandb
from tqdm import tqdm
from utils.log_utils import log_values
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from eval import evaluate
from utils.functions import accuracy


def train(model, optimizer, train_dataset, val_dataset, tb_logger, opts):
    training_dataloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=16)
    step = 0
    model.train()
    for epoch in tqdm(range(opts.n_epochs)):
        running_loss = 0.0
        running_accu = 0.0
        for train_batch in training_dataloader:
            train_batch.to(opts.device)
            start_time = time.time()

            optimizer.zero_grad()
            train_batch_out, _ = model(train_batch, compute_embeddings=False)
            
            # regresssion loss(MSE)
            # train_batch_loss = F.mse_loss(train_batch_out, train_batch.y)

            # classification code by binarizing score
            # train_batch_y = torch.tensor(train_batch.y > 0.0)
            # train_batch_y.to(opts.device)
            # train_batch_y = train_batch_y.type(torch.cuda.FloatTensor)
            # train_batch_loss = F.binary_cross_entropy(train_batch_out, train_batch_y)
            # train_batch_accu = accuracy(output=train_batch_out, target=train_batch_y, threshold=0.5)
            # classification code

            # node classification by binary labels
            train_batch_loss = F.binary_cross_entropy(train_batch_out, train_batch.y)
            train_batch_accu = accuracy(output=train_batch_out, target=train_batch.y, threshold=0.5)
            train_batch_loss.backward()
            optimizer.step()
            
            running_accu += train_batch_accu.item()
            running_loss += train_batch_loss.item()

        # train loss at each epoch
        train_loss = running_loss / len(training_dataloader)

        # train accuracy at each epoch
        train_accu = running_accu / len(training_dataloader) * 100

        # val loss and accuracy at each epoch
        val_loss, val_accu = evaluate(model, val_dataset, opts)

        # Logging
        if step % int(opts.log_step) == 0:
            # wandb logging
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accu": train_accu,
                "val_accu": val_accu
            })

            # Tensorboard logging
            log_values(epoch, step, tb_logger, opts, train_loss, val_loss)

        step += 1
        epoch_duration = time.time() - start_time
        print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    return "Training Completed!"
