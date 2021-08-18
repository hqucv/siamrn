from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamban.core.config import cfg
from siamban.tracker.siamban_tracker import SiamBANTracker
from siamban.tracker.siambanmulti_tracker import SiamBANMultiTracker

TRACKS = {
          'SiamBANTracker': SiamBANTracker,
          'SiamRPNMultiTracker': SiamBANMultiTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
