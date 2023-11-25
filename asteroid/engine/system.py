import torch
import csv
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from asteroid_filterbanks import make_enc_dec
from ..utils import flatten_dict


class ssl_loss(torch.nn.Module):
    def __init__(self,ssl_model):
        super().__init__()
        self.ssl_model = ssl_model
    def forward(self,input,est_targets):
        est_targets_1,est_targets_2 = est_targets.split([1,1],dim=1)
        cancat = torch.cat((input,est_targets_1.squeeze(1),est_targets_2.squeeze(1)),dim=1)
        loss = self.ssl_model(cancat)
        # print(loss.size())
        return -torch.mean(loss.squeeze(0))
class System(pl.LightningModule):
    """Base class for deep learning systems.
    Contains a model, an optimizer, a loss function, training and validation
    dataloaders and learning rate scheduler.

    Note that by default, any PyTorch-Lightning hooks are *not* passed to the model.
    If you want to use Lightning hooks, add the hooks to a subclass::

        class MySystem(System):
            def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
                return self.model.on_train_batch_start(batch, batch_idx, dataloader_idx)

    Args:
        model (torch.nn.Module): Instance of model.
        optimizer (torch.optim.Optimizer): Instance or list of optimizers.
        loss_func (callable): Loss function with signature
            (est_targets, targets).
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Instance, or list
            of learning rate schedulers. Also supports dict or list of dict as
            ``{"interval": "step", "scheduler": sched}`` where ``interval=="step"``
            for step-wise schedulers and ``interval=="epoch"`` for classical ones.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.

    .. note:: By default, ``training_step`` (used by ``pytorch-lightning`` in the
        training loop) and ``validation_step`` (used for the validation loop)
        share ``common_step``. If you want different behavior for the training
        loop and the validation loop, overwrite both ``training_step`` and
        ``validation_step`` instead.

    For more info on its methods, properties and hooks, have a look at lightning's docs:
    https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#lightningmodule-api
    """

    default_monitor: str = "val_loss"

    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        # ssl_predict,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        # self.ssl_predict = ssl_predict
        # self.ssl_predict = ssl_loss(ssl_predict)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.csv_train_name =  config["main_args"]["exp_dir"] + "/train_csv.csv"
        self.csv_dev_name = config["main_args"]["exp_dir"] + "/dev_csv.csv"
        self.conflict_dir_name = config["main_args"]["exp_dir"] + "/conflict_statistic.csv"
        self.config = {} if config is None else config
        self.mse_loss = torch.nn.MSELoss()
        self.cossim = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        # self.scaler = torch.cuda.amp.GradScaler()
        # self.automatic_optimization = False
        # Save lightning's AttributeDict under self.hparams
        self.save_hyperparameters(self.config_to_hparams(self.config))

    def forward(self, *args, **kwargs):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.model(*args, **kwargs)

    def cos_eval(self,estimated,target,short_time=False):
        est_targets_1,est_targets_2 = estimated.split([1,1],dim=1)
        targets_1,target_2 = target.split([1,1],dim=1)
        cos = torch.nn.CosineSimilarity(dim=-1,eps=1e-08)
        mse = torch.nn.MSELoss()
        if short_time == False:
            # print(torch.acos(cos(est_targets_1,est_targets_2)))
            # print(torch.acos(cos(targets_1,target_2)))
            # print(torch.tan(torch.acos(cos(est_targets_1,est_targets_2))-torch.acos(cos(targets_1,target_2)))**2)
            # print(20*torch.log10(torch.tan(torch.acos(cos(est_targets_1,est_targets_2))-torch.acos(cos(targets_1,target_2)))**2))
            # sisnr +sisnr
            # return 20*torch.log10(torch.abs(torch.tan(torch.acos(cos(est_targets_1,est_targets_2))-torch.acos(cos(targets_1,target_2)))+0.00005)**2)
            # sisnr +mse (cos**2 +cos**2)
            # return mse(cos(est_targets_1**2,est_targets_2**2),cos(targets_1**2,target_2**2))
            # sisnr +mse (cos +cos)
            # torch.acos(cos(est_targets_1,est_targets_2))-torch.acos(cos(targets_1,target_2))
            # print(torch.cos( torch.acos(cos(est_targets_1,est_targets_2))- torch.acos(cos(targets_1,target_2))  ))
            return torch.pow(1-torch.mean(torch.cos( torch.acos(cos(est_targets_1,est_targets_2))- torch.acos(cos(targets_1,target_2))  )),2)
        else:
            n = int(8000 * 5 / 4000)
            b , t, f = est_targets_1.size() # 4,1,40000
            new_f = int(f/n)
            est_targets_1,est_targets_2 = est_targets_1.reshape(b*n,t,new_f),est_targets_2.reshape(b*n,t,new_f)
            targets_1,target_2 = targets_1.reshape(b*n,t,new_f),target_2.reshape(b*n,t,new_f)
            return mse(cos(est_targets_1,est_targets_2),cos(targets_1,target_2))
    def sisnr_eval(self,est_targets,targets,short_time=False):
        if short_time == False:
            return self.loss_func(est_targets, targets)
        else:
            n = int(8000 * 5 / 4000)
            est_targets,targets = torch.cat(est_targets.chunk(n,dim=-1),dim=0),torch.cat(targets.chunk(n,dim=-1),dim=0)
            return self.loss_func(est_targets, targets)

    def common_step(self, batch, batch_nb, train=True):
        """Common forward step between training and validation.

        The function of this method is to unpack the data given by the loader,
        forward the batch through the model and compute the loss.
        Pytorch-lightning handles all the rest.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
            train (bool): Whether in training mode. Needed only if the training
                and validation steps are fundamentally different, otherwise,
                pytorch-lightning handles the usual differences.

        Returns:
            :class:`torch.Tensor` : The loss value on this batch.

        .. note::
            This is typically the method to overwrite when subclassing
            ``System``. If the training and validation steps are somehow
            different (except for ``loss.backward()`` and ``optimzer.step()``),
            the argument ``train`` can be used to switch behavior.
            Otherwise, ``training_step`` and ``validation_step`` can be overwriten.
        """
        inputs, targets, noise = batch
        enhanced_wav,separated_wav = self(inputs)
        if train== False:
            return self.sisnr_eval(enhanced_wav, targets,short_time=False)
        else:
            tgt_1,tgt_2 = torch.split(targets,[1,1],dim=1)
            tgt_1_mid = tgt_1 + noise.unsqueeze(1)
            tgt_2_mid = tgt_2 + noise.unsqueeze(1)
 
            tgt_mid = torch.cat((tgt_1_mid,tgt_2_mid),dim=1)
            absmax = torch.max(torch.abs(tgt_mid))
            if absmax > 1:
                tgt_mid = tgt_mid / absmax
            return 0.5*self.loss_func(separated_wav, tgt_mid) + 0.5*self.loss_func(enhanced_wav, targets)
          

    def training_step(self, batch, batch_nb):
        """Pass data through the model and compute the loss.

        Backprop is **not** performed (meaning PL will do it for you).

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            torch.Tensor, the value of the loss.
        """
        # print(self.trainer.current_epoch)
        # print(self.trainer.global_step//3475)
        # for (name,param) in self.model.named_parameters():
        #     print(name,param.requires_grad)
        # if self.trainer.current_epoch >= 10:
        #    for (name,param) in self.model.named_parameters():
        #        if "er_enh" in name:
        #            param.requires_grad = False
        #        else:
        #            pass
        # for (name,param) in self.model.named_parameters():
        #     print(name,param.requires_grad)
        loss_est = self.common_step(batch, batch_nb, train=True)
        self.log("loss", loss_est, logger=True)
        return loss_est
        # path = self.csv_train_name
        # with open(path,"a+") as f:
        #     csv.write = csv.writer(f)
        #     data_row=[self.optimizer.state_dict()['param_groups'][0]['lr'],loss_real,loss_est]
        #     csv.write.writerow(data_row)
        # self.log("loss", loss_est, logger=True)
        # print("lr",self.optimizer.state_dict()['param_groups'][0]['lr'])
        # return loss_est
        # inputs, targets = batch
        # est_targets = self(inputs)
        # est_targets_1,est_targets_2 = est_targets.split([1,1],dim=1).clone()
        # cancat = torch.cat((inputs,est_targets.split([1,1],dim=1)[0].squeeze(1),est_targets.split([1,1],dim=1)[1].squeeze(1)),dim=1)
        # loss_2 = -self.ssl_predict(cancat)
        # print(type(loss_2))
        # path = self.csv_train_name
        # with open(path,"a+") as f:
        #     csv.write = csv.writer(f)
        #     data_row=[self.optimizer.state_dict()['param_groups'][0]['lr'],self.loss_func(est_targets, targets),loss_2]
        #     csv.write.writerow(data_row)
        # self.log("loss", loss_2, logger=True)
        # return torch.mean(loss_2.squeeze(0))

    def validation_step(self, batch, batch_nb):
        """Need to overwrite PL validation_step to do validation.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
        """
        loss_est = self.common_step(batch, batch_nb, train=False)

        # path = self.csv_dev_name
        # with open(path,"a+") as f:
        #     csv.write = csv.writer(f)
        #     data_row=[loss_real,loss_est]
        #     csv.write.writerow(data_row)
        # loss = self.common_step(batch, batch_nb, train=False)
        self.log("val_loss", loss_est, on_epoch=True, prog_bar=True)
        # inputs, targets = batch
        # est_targets = self(inputs)
        # loss = self.loss_func(est_targets, targets)
        # self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    # def on_epoch_start(self):

    def on_validation_epoch_end(self):
        """Log hp_metric to tensorboard for hparams selection."""
        # print(self.trainer.callback_metrics)
        # print(self.trainer.callbacks)
        # if self.trainer.current_epoch ==10:
            # print(self.trainer.callbacks[1]) # lr_moniter
            # print(self.trainer.callbacks[4]) # lr_moniter
        #    self.trainer.callbacks[0].change_best_score(-0.1)
        #    self.trainer.callbacks[4].change_best_model_score(-0.1)
        #    self.scheduler[0].best = self.trainer.callback_metrics.get("val_loss", None)
            # self.scheduler[0].best=0
        hp_metric = self.trainer.callback_metrics.get("val_loss", None)

        print("***",self.optimizer.state_dict()['param_groups'][0]['lr'])
        # self.scheduler[0].step(hp_metric)
        # self.save_ckpt()
        # save_dir = "/home/dangsx/asteroid/egs/librimix/dilatedDPRNN/exp/diff_e2e/new/enh_sep_2to6_minus_GM/checkpoints/"
        # name = str(self.trainer.current_epoch)
        # path = f"{save_dir}/{name}.ckpt"
        # self.trainer.save_checkpoint(path, False)
        if hp_metric is not None:
            self.trainer.logger.log_metrics({"hp_metric": hp_metric}, step=self.trainer.global_step)

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer
        # first_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1, total_iters=5,last_epoch=5)

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
