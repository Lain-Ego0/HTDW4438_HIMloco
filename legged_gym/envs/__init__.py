from .base.legged_robot import LeggedRobot
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO
from .htdw_4438.htdw_4438_config import Htdw4438Cfg, Htdw4438CfgPPO
from .htdw_4438_v2.htdw_4438_v2_config import Htdw4438V2Cfg, Htdw4438V2CfgPPO
from .opendoge.opendoge_config import OpendogeCfg, OpendogeCfgPPO

from legged_gym.utils.task_registry import task_registry

task_registry.register("a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO())
task_registry.register("go1", LeggedRobot, Go1RoughCfg(), Go1RoughCfgPPO())
task_registry.register("htdw_4438", LeggedRobot, Htdw4438Cfg(), Htdw4438CfgPPO())
task_registry.register("htdw_4438_v2", LeggedRobot, Htdw4438V2Cfg(), Htdw4438V2CfgPPO())
task_registry.register("opendoge", LeggedRobot, OpendogeCfg(), OpendogeCfgPPO())
