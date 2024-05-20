# DanLing
# Copyright (C) 2022-Present  DanLing

# This program is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from typing import Optional


class BaseX:
    alphabet: str

    def __init__(self, alphabet: Optional[str] = None) -> None:
        if alphabet is not None:
            self.alphabet = alphabet
        self.num_alphabets = len(self.alphabet)
        self.alphabet_dict = {a: i for i, a in enumerate(self.alphabet)}

    def encode(self, decimal: int) -> str:
        if decimal == 0:
            return self.alphabet[0]
        encoded = ""
        while decimal != 0:
            encoded = self.alphabet[decimal % self.num_alphabets] + encoded
            decimal //= self.num_alphabets
        return encoded

    def decode(self, string: str) -> int:
        decoded = 0
        for i, a in enumerate(reversed(string)):
            decoded += (self.num_alphabets**i) * self.alphabet_dict[a]
        return decoded


class Base58(BaseX):
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


class Base62(BaseX):
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class Base64(BaseX):
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/"


base58 = Base58()
base62 = Base62()
base64 = Base64()
