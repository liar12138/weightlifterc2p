"""ZKC technique barbell (ZKCB-T10) object constants."""

from pathlib import Path

import mujoco
import random

from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from functools import partial   # 加到文件顶部 import

##
# MJCF.
##

_OBJECT_ROOT = Path(__file__).resolve().parent
BARBELL_XML = _OBJECT_ROOT / "xmls" / "olympic_barbell.xml"
assert BARBELL_XML.exists()
BARBELL_RACE_XML = _OBJECT_ROOT / "xmls" / "olympic_barbell_race.xml"
assert BARBELL_RACE_XML.exists()


# def get_spec() -> mujoco.MjSpec:
#     spec = mujoco.MjSpec.from_file(str(BARBELL_XML))
#     diameter = random.uniform(0.024, 0.031)
#     radius = diameter / 2.0

#     shaft = spec.geom("shaft")

#     shaft_collision = spec.geom("shaft_collision")

#     shaft.size[0] = radius
#     shaft_collision.size[0] = radius

#     print(
#         f"[Barbell Randomization] "
#         f"diameter={diameter * 1000:.2f} mm, "
#         f"radius={radius * 1000:.2f} mm"
#     )

#     return spec
##
# Keyframe / initial state.
##

# Plate radius is 0.225 m; keep the bar clear of the ground at spawn.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  rot=(1.0, 0.0, 0.0, 0.0),
  lin_vel=(0.0, 0.0, 0.0),
  ang_vel=(0.0, 0.0, 0.0),
)

##
# Collision config.
##

SHAFT_COLLISION_PATTERN = r"^shaft_collision$"
PLATE_COLLISION_PATTERN = r"^plate_.*_collision$"

# High friction on the shaft for stable dual-arm / dexterous grasping.
# Plates keep condim=1 to limit contact count; shaft uses condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=[".*_collision"],
  condim={SHAFT_COLLISION_PATTERN: 3, ".*_collision": 1},
  priority={SHAFT_COLLISION_PATTERN: 1},
  friction={
    SHAFT_COLLISION_PATTERN: (1.2, 0.005, 0.0001),
    PLATE_COLLISION_PATTERN: (0.8,),
    ".*_collision": (0.6,),
  },
)

# Match robot configs that disable self-collision via contype=0 / conaffinity=1.
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=[".*_collision"],
  contype=0,
  conaffinity=1,
  condim={SHAFT_COLLISION_PATTERN: 3, ".*_collision": 1},
  priority={SHAFT_COLLISION_PATTERN: 1},
  friction={
    SHAFT_COLLISION_PATTERN: (1.2, 0.005, 0.0001),
    PLATE_COLLISION_PATTERN: (0.8,),
    ".*_collision": (0.6,),
  },
)



# def get_barbell_cfg() -> EntityCfg:
#   """Get a fresh Olympic barbell configuration instance."""
#   return EntityCfg(
#     init_state=INIT_STATE,
#     collisions=(FULL_COLLISION,),
#     spec_fn=get_spec,
#   )


# 国际标准比赛（IWF / 奥林匹克）质量，单位 kg
MEN_COMPETITION_BAR_MASS = 20.0    # 男子比赛杆：杆身 + 两侧套筒 合计 20 kg
WOMEN_COMPETITION_BAR_MASS = 15.0  # 女子比赛杆：合计 15 kg
# 标准比赛杠铃片（单边单片常见配重）
COMPETITION_PLATE_MASSES = (25.0, 20.0, 15.0, 10.0, 5.0, 2.5)


def _set_bar_mass(spec: mujoco.MjSpec, bar_mass: float) -> None:
    """把整杆质量（shaft + 两侧 sleeve）按几何体积比例分配到各碰撞 geom。

    bar_mass 指“整根杆”的质量（IWF 男子 20 kg / 女子 15 kg）。
    杆身与套筒都是沿 Y 的圆柱，geom.size = (半径, 半长)。
    """
    shaft = spec.geom("shaft_collision")
    sleeve_l = spec.geom("sleeve_left_collision")

    r_shaft, half_shaft = shaft.size[0], shaft.size[1]
    r_sleeve, half_sleeve = sleeve_l.size[0], sleeve_l.size[1]

    vol_shaft = r_shaft**2 * 2.0 * half_shaft
    vol_sleeve = r_sleeve**2 * 2.0 * half_sleeve
    vol_total = vol_shaft + 2.0 * vol_sleeve

    shaft.mass = bar_mass * vol_shaft / vol_total
    sleeve_mass = bar_mass * vol_sleeve / vol_total
    sleeve_l.mass = sleeve_mass
    spec.geom("sleeve_right_collision").mass = sleeve_mass


def get_spec(
    plate_mass: float | None = None,
    bar_mass: float | None = None,
) -> mujoco.MjSpec:
    """Build the Olympic barbell spec.

    Args:
        plate_mass: 单边单片杠铃片的质量（kg）；None 时使用 XML 原始质量。
        bar_mass: 整根杆（shaft + 两侧 sleeve）的质量（kg）；
                  None 时使用 XML 原始质量。
    """
    spec = mujoco.MjSpec.from_file(str(BARBELL_XML))
    diameter = random.uniform(0.024, 0.031)
    radius = diameter / 2.0

    shaft = spec.geom("shaft")
    shaft_collision = spec.geom("shaft_collision")

    shaft.size[0] = radius
    shaft_collision.size[0] = radius

    plate_diameter = random.uniform(0.35, 0.45)
    plate_radius = plate_diameter / 2.0

    spec.geom("plate_left").size[0] = plate_radius
    spec.geom("plate_left_collision").size[0] = plate_radius

    spec.geom("plate_right").size[0] = plate_radius
    spec.geom("plate_right_collision").size[0] = plate_radius
    if bar_mass is not None:
        # 国际标准：给杆分配真实质量，而不是把杆设成 0 质量
        _set_bar_mass(spec, bar_mass)
    if plate_mass is not None:
        spec.geom("plate_left_collision").mass = plate_mass
        spec.geom("plate_right_collision").mass = plate_mass

    # print(
    #     f"[Barbell Randomization] "
    #     f"diameter={diameter * 1000:.2f} mm, "
    #     f"radius={radius * 1000:.2f} mm"
    #     + (f", plate_mass={plate_mass} kg" if plate_mass is not None else "")
    #     + (f", bar_mass={bar_mass} kg" if bar_mass is not None else "")
    # )

    return spec

##
# Final config.
##

BARBELL_CFG = EntityCfg(
  init_state=INIT_STATE,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
)


def get_barbell_cfg(
    plate_mass: float | None = None,
    bar_mass: float | None = None,
) -> EntityCfg:
  """Get a fresh Olympic barbell configuration instance.

  Args:
      plate_mass: 单边单片杠铃片质量（kg），总重 = bar_mass + 2 × plate_mass；
                  为 None（默认）时使用 XML 原始质量（杆+片都保留原值）。
      bar_mass: 整根杆（shaft + 两侧 sleeve）的质量（kg），
                如 IWF 男子杆传 MEN_COMPETITION_BAR_MASS = 20.0；
                为 None（默认）时使用 XML 原始质量。
  """
  spec_fn = partial(get_spec, plate_mass=plate_mass, bar_mass=bar_mass)
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=spec_fn,
  )

def get_race_spec() -> mujoco.MjSpec:
  """Race barbell spec（10kg 质量等已写入 olympic_barbell_race.xml）。"""
  return mujoco.MjSpec.from_file(str(BARBELL_RACE_XML))


def get_barbell_race_cfg() -> EntityCfg:
  """Get a fresh race barbell (olympic_barbell_race.xml) configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_race_spec,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity
  from mjlab.utils.spec_config import LightCfg, MaterialCfg, TextureCfg

  barbell = Entity(BARBELL_CFG)
  spec = barbell.spec

  # Checkerboard ground (same defaults as TerrainImporter plane).
  TextureCfg(
    name="groundplane",
    type="2d",
    builtin="checker",
    mark="edge",
    rgb1=(0.2, 0.3, 0.4),
    rgb2=(0.1, 0.2, 0.3),
    markrgb=(0.8, 0.8, 0.8),
    width=300,
    height=300,
  ).edit_spec(spec)
  MaterialCfg(
    name="groundplane",
    texuniform=True,
    texrepeat=(4, 4),
    reflectance=0.2,
    texture="groundplane",
  ).edit_spec(spec)
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0, 0, 0.01),
    material="groundplane",
  )

  # Key lights for native MuJoCo viewer.
  LightCfg(
    name="sun",
    type="directional",
    pos=(0.0, 0.0, 2.5),
    dir=(0.0, 0.0, -1.0),
    castshadow=True,
  ).edit_spec(spec)
  LightCfg(
    name="fill",
    type="spot",
    pos=(1.5, 1.5, 2.0),
    dir=(-0.5, -0.5, -1.0),
    castshadow=False,
  ).edit_spec(spec)

  mj_model = spec.compile()
  print(
    f"nq={mj_model.nq}, nv={mj_model.nv}, nbody={mj_model.nbody}, "
    f"ngeom={mj_model.ngeom}, fixed={barbell.is_fixed_base}, "
    f"articulated={barbell.is_articulated}, actuated={barbell.is_actuated}"
  )
  viewer.launch(mj_model)
