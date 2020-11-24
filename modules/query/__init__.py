from modules.query.dn4 import *
from modules.query.mn4 import *
from modules.query.dmn4 import *
from modules.query.relation import *
from modules.query.protonet import *

import modules.registry as registry

def make_query(in_channels, cfg):
    return registry.Query[cfg.model.query](in_channels, cfg)
