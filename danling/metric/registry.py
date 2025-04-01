# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

from __future__ import annotations

from chanfig import Registry as Registry_

from .factory import binary_metrics, multiclass_metrics, multilabel_metrics, regression_metrics
from .metrics import Metrics


class Registry(Registry_):
    case_sensitive = False

    def build(
        self,
        type: str,
        num_labels: int | None = None,
        num_classes: int | None = None,
        num_outputs: int | None = None,
        **kwargs,
    ) -> Metrics:
        type = type.lower()
        if type == "multilabel":
            return self.init(self.lookup(type), num_labels=num_labels, **kwargs)
        if type == "multiclass":
            num_classes = num_classes or num_labels
            return self.init(self.lookup(type), num_classes=num_classes, **kwargs)
        if type == "regression":
            num_outputs = num_outputs or num_labels
            return self.init(self.lookup(type), num_outputs=num_outputs, **kwargs)
        return self.init(self.lookup(type), **kwargs)


METRICS = Registry(key="type")
METRICS.register(binary_metrics, "binary")
METRICS.register(multiclass_metrics, "multiclass")
METRICS.register(multilabel_metrics, "multilabel")
METRICS.register(regression_metrics, "regression")
