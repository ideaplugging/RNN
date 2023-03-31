from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from classification.utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class MyEngine(Engine):

    def __init__(self, func, model, crit, optimizer, config):
        # Ignite Engine does not have objects in below lines.
        # Thus, we assign class variables to access these objects, during the procedure.
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func) # Ignite Engine only need function to run.

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod # training func
    def train(engine, mini_batch): # tuple in mini-batch
        # You have to reset the gradients of all models parameters
        # before to take another step in gradient descent.
        engine.model.train() # Because we assign models as class variable, we can easily access to fit.
        engine.optimizer.zero_grad() # Being called back per iteration, zero_grad() declare.

        x, y = mini_batch
        x, y = x.to(engine.device), y.to(engine.device) # model과 같은 device로 보냄

        # Take feed-forward
        y_hat = engine.model(x) # |x| = (bs, 784) , |y_hat| = (bs, 10)

        loss = engine.crit(y_hat, y) # scalar
        loss.backward() # backpropagation

        # Calculate accuracy only if 'y' is LongTensor.
        # which means that 'y' is one-hot representation.

        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor): # True = classification task
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0)) # y.size(0) = batch size
        else: # False = regression task
            accuracy = 0

        # 학습의 보조 지표로 사용. 학습의 안정성을 알아볼 수 있음
        p_norm = float(get_parameter_norm(engine.model.parameters())) # parameter의 L2 norm, 학습이 진행될 수록 커져야 함 = 파라미터가 업데이트 되고 있구나
        g_norm = float(get_grad_norm(engine.model.parameters())) # loss의 L2 norm, gradient의 크기가 크다 = 많은 걸 배우고 있다 = 조금만 배워도 gradient가 크게 변한다. = loss surface가 얼마나 가파른지를 알 수 있음

        if engine.config.max_grad > 0:
            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad,
                norm_type=2,
            )

        # Take a step of gradient descent.

        engine.optimizer.step()

        return { # code의 가독성을 위해서 dic를 사용
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    @staticmethod # validation func
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch
            x, y = x.to(engine.device), y.to(engine.device)

            y_hat = engine.model(x)

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
        }

    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE): # visualization of a learning.
        # Attaching would be repeated for several metrics.
        # Thus, we can reduce the reeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine, # Being called back per iteration
                metric_name,
            )

        training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|']

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name) # 계산해서 가지고만 있는 상태

        # If the verbosity is set, progress bar would be shown for mini-batch iterations.
        # Without ignite, you can use tqdm to implement progress bar.

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        # If the verbosity is set, statistics would be shown after each epoch.
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} accuracy={:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))

        validation_metric_names = ['loss', 'accuracy']

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        # Do same things for validation engine.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        # If the verbosity is set, statistics would be shown after each epoch.
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4e}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.best_loss,
                ))

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss: # If current epoch returns the lower validation loss,
            engine.best_loss = loss # Update the lowest validation loss.
            engine.best_model = deepcopy(engine.model.state_dict()) # Update the best models weights.

    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'models': engine.best_model,
                'config': config,
                **kwargs
            }, config.model_fn
        )

class Trainer():

    def __init__(self, config):
        self.config = config

    def train(
            self,
            model, crit, optimizer,
            train_loader, valid_loader
    ):
        train_engine = MyEngine(
            MyEngine.train, # iteration 마다 호출됨
            model, crit, optimizer, self.config
        )
        validation_engine = MyEngine(
            MyEngine.validate, # iteration 마다 호출됨
            model, crit, optimizer, self.config
        )

        MyEngine.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader): # training engine이 끝난 후 validation engine이 호출되도록 연결하는 함수
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            run_validation, # function
            validation_engine, valid_loader, # argumnets
        )

        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            MyEngine.check_best, # function
        )

        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            MyEngine.save_model, # function
            train_engine, self.config # arguments
        )

        train_engine.run( # 실제 학습 실행
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        model.load_state_dict(validation_engine.best_model)

        return model