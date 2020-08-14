import sys
import torch
from torchnet.meter import AverageValueMeter
import datetime
import tensorflow as tf
from tensorflow import summary
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
writer = SummaryWriter('logs/exp1 - se_resnext101_32x4d,epoch=10,activation_func=sigmoid,batch_size=1')



class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, ep, batch):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        metric_values = np.empty(1)
        ind=0

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y, _ in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)



                # update metrics logs
                j=0
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    if j==1:
                        metric_values = np.append(metric_values, [metric_value])
                    metrics_meters[metric_fn.__name__].add(metric_value)
                    j+=1
                metrics_logs = {k+'mean': v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)
                ind+=1
                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        metrics_logs.update({'fscore_median': np.median(metric_values)})
        metrics_logs.update({'fscore_25th': np.quantile(metric_values, q=0.25)})
        metrics_logs.update({'fscore_75th': np.quantile(metric_values, q=0.75)})
        logs.update(metrics_logs)
        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        #prediction = torch.div(prediction, 255) #an attempt to match the prediction image value range with label value range
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)

            #prediction = torch.div(prediction, 255) #an attempt to match the prediction image value range with label value range
            loss = self.loss(prediction, y)
        return loss, prediction