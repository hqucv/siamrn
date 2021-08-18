from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamban.models.head.ban import UPChannelBAN, DepthwiseBAN, MultiBAN
from siamban.models.head.dis_detector import MultiDetector


BANS = {
        'UPChannelBAN': UPChannelBAN,
        'DepthwiseBAN': DepthwiseBAN,
        'MultiBAN': MultiBAN
       }

DECT = {
        'MultiDect': MultiDetector
}

def get_ban_head(name, **kwargs):
    return BANS[name](**kwargs)

def get_dect_head(name, **kwargs):
    return DECT[name](**kwargs)
