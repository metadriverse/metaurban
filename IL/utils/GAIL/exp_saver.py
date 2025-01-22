import copy
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tensorboardX import SummaryWriter


def _format(**kwargs):
    result = list()

    for k, v in kwargs.items():
        if isinstance(v, float) or isinstance(v, np.float32):
            result.append('%s: %.2f' % (k, v))
        else:
            result.append('%s: %s' % (k, v))

    return '\t'.join(result)


class Experiment(object):
    def init(self, log_dir):
        """
        This MUST be called.
        """
        self._log = logger
        self.epoch = 0
        self.scalars = OrderedDict()

        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # for i in self._log._handlers:
        #     self._log.remove(i)

        self._writer_train = SummaryWriter(str(self.log_dir / 'train'))
        self._writer_val = SummaryWriter(str(self.log_dir / 'val'))
        self._log.add(
            str(self.log_dir / 'log.txt'),
            format='{time:MM/DD/YY HH:mm:ss} {level}\t{message}')

        # Functions.
        self.debug = self._log.debug
        self.info = lambda **kwargs: self._log.info(_format(**kwargs))

    def load_config(self, model_path):
        log_dir = Path(model_path).parent

        with open(str(log_dir / 'config.json'), 'r') as f:
            return json.load(f)

    def save_config(self, config_dict):
        def _process(x):
            for key, val in x.items():
                if isinstance(val, dict):
                    _process(val)
                elif not isinstance(val, float) and not isinstance(val, int):
                    x[key] = str(val)

        config = copy.deepcopy(config_dict)

        _process(config)

        with open(str(self.log_dir / 'config.json'), 'w+') as f:
            json.dump(config, f, indent=4, sort_keys=True)

    def scalar(self, is_train=True, **kwargs):
        for k, v in sorted(kwargs.items()):
            key = (is_train, k)

            if key not in self.scalars:
                self.scalars[key] = list()

            self.scalars[key].append(v)

    def end_epoch(self, epoch, net=None):
        self.info(Epoch=epoch)
        for (is_train, k), v in self.scalars.items():
            info = OrderedDict()
            info['%s_%s' % ('Train' if is_train else 'Eval', k)] = np.mean(v)
            info['std'] = np.std(v, dtype=np.float32)
            info['min'] = np.min(v)
            info['max'] = np.max(v)

            self.info(**info)

            if is_train:
                self._writer_train.add_scalar(k, np.mean(v), epoch)
            else:
                self._writer_val.add_scalar(k, np.mean(v), epoch)

        self.scalars.clear()

        if net is not None:
            if self.epoch % 10 == 0:
                torch.save(net.state_dict(), str(self.log_dir / ('model_%03d.t7' % self.epoch)))

            torch.save(net.state_dict(), str(self.log_dir / 'latest.t7'))

        self.epoch += 1