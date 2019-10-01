import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from utils.functions import accuracy


def evaluate(model, data, opts):
    model.eval()
    val_dataloader = DataLoader(dataset=data, batch_size=opts.batch_size, shuffle=False, num_workers=16)
    with torch.no_grad():
        running_loss = 0.0
        running_accu = 0.0
        for val_batch in val_dataloader:
            val_batch.to(opts.device)
            val_batch_out, _ = model(val_batch, compute_embeddings=False)

            # regression loss
            # loss = F.mse_loss(out_val_batch, val_batch.y)

            # classifcation by binarizing score
            # val_batch_y = torch.tensor(val_batch.y > 0.0)
            # val_batch_y.to(opts.device)
            # val_batch_y = val_batch_y.type(torch.cuda.FloatTensor)
            # val_batch_loss = F.binary_cross_entropy(val_batch_out, val_batch_y)
            # val_batch_accu = accuracy(output=val_batch_out, target=val_batch_y, threshold=0.5)

            # classification using one aprox sol with binary labels
            val_batch_loss = F.binary_cross_entropy(val_batch_out, val_batch.y)
            val_batch_accu = accuracy(output=val_batch_out, target=val_batch.y, threshold=0.5)
            running_loss += val_batch_loss.item()
            running_accu += val_batch_accu.item()

    return running_loss / len(val_dataloader), running_accu / len(val_dataloader) * 100


