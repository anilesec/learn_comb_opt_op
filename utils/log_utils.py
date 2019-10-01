def log_values(epoch, step, tb_logger, opts, train_loss, val_loss):
    # log values to screen
    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}'
    print(log.format(epoch, train_loss, val_loss))

    # log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value("train_loss", train_loss, step)
        tb_logger.log_value("val_loss", val_loss, step)



