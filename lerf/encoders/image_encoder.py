# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod, abstractproperty
from dataclasses import dataclass, field
from typing import Type

import torch
# from nerfstudio.configs import base_config as cfg
from torch import nn


# @dataclass
# class BaseImageEncoderConfig(cfg.InstantiateConfig):
#     _target: Type = field(default_factory=lambda: BaseImageEncoder)


class BaseImageEncoder(nn.Module):
    @abstractproperty
    def name(self) -> str:
        """
        returns the name of the encoder
        """

    @abstractproperty
    def embedding_dim(self) -> int:
        """
        returns the dimension of the embeddings
        """

    @abstractmethod
    def encode_image(self, input: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of input images, return their encodings
        """

    @abstractmethod
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        """
        Given a batch of embeddings, return the relevancy to the given positive id
        """
