# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .trans2ray import build


def build_model(args):
    return build(args)
