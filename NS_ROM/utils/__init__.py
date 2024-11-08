# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from .error_analysis import error_analysis_pinball
from .speedup_analysis import speedup_analysis_pinball

__all__ = [
    "error_analysis_pinball",
    "speedup_analysis_pinball"
]
