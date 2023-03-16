# [DanLing](https://danling.org)

## Introduction

DanLing (丹灵) is a high-level library to help with running neural networks flexibly and transparently.

DanLing is meant to be a scaffold for experienced researchers and engineers who know how to define a training loop, but are bored of writing the same boilerplate code, such as DDP, logging, checkpointing, etc., over and over again.

Therefore, DanLing does not feature complex Runner designs with many pre-defined methods and complicated hooks.
Instead, the Runner of DanLing just initialise the essential parts for you, and you can do whatever you want, however you want.

Although many attributes and properties are pre-defined and are expected to be used in DanLing, you have full control over your code.

DanLing also provides some utilities, such as [Registry][danling.Registry], [NestedTensor][danling.NestedTensor], [catch][danling.utils.catch], etc.

## Installation

Install the most recent stable version on pypi:

```shell
pip install danling
```

Install the latest version from source:

```shell
pip install git+https://github.com/ZhiyuanChen/DanLing
```

It works the way it should have worked.

## License

DanLing is multi-licensed under the following licenses:

- Unlicense
- GNU GPL 2.0 (or any later version)
- MIT
- Apache 2.0
- BSD 2-Clause
- BSD 3-Clause
- BSD 4-Clause

You can choose any (one or more) of them if you use this work.

`SPDX-License-Identifier: Unlicense OR GPL-2.0-or-later OR MIT OR Apache-2.0 OR BSD-2-Clause OR BSD-3-Clause OR BSD-4-Clause`
