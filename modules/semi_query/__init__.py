from modules.semi_query.dn4 import *
from modules.semi_query.mn4 import *
import modules.registry as registry

def make_query(in_channels, cfg):
    return registry.SemiQuery[cfg.model.query](in_channels, cfg)

