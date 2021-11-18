from typing import List
from tqdm import tqdm
import pandas as pd
class Trainwrapper:

    def __init__(self, log_values: List):
        self.log_values = log_values
        

    def wrap_train(self, epochs, train_loader, train_step_fun, train_hyper_fun = None, hyper_update_step=2):
        epoch_losses = []
        pbar = tqdm(range(epochs))
        epoch_losses = pd.DataFrame(columns=["epoch"] + self.log_values)
        for epoch in pbar:
            batch_loss = pd.DataFrame(columns=self.log_values)
            for batch_id, (data, target) in enumerate(train_loader):
                # forward pass
                loss = train_step_fun(data, target)
                # logging
                batch_loss = batch_loss.append(pd.DataFrame([loss], columns=self.log_values))

            current_epoch_loss = batch_loss.mean(axis=0)
            description = " ".join([f"{name}:{round(current_epoch_loss[name], 2)}" for name in batch_loss.columns])
            pbar.set_description(description)
            epoch_losses = epoch_losses.append(pd.DataFrame(pd.concat([pd.Series(epoch, index=["epoch"]), current_epoch_loss])).T)

            # train hyperparameters
            if train_hyper_fun and (epoch % hyper_update_step == 0):
                train_hyper_fun(train_loader)

        return epoch_losses