"""Decide how every texture is made and at what size, within the atlas budget.

    python3 plan.py <src_dir> <plan.json>

Block and item textures share Minecraft's block atlas. For the target GPU
(10-12 GB VRAM) the pack aims at the standard 16384 x 16384 atlas, ~268M texels;
the plan keeps total output under BUDGET to leave room for packing and other packs.
Iris allocates matching normal and specular atlases, so this also bounds VRAM.

Tiers (output pixels per 16 source pixels):
  hero      256  most-seen full blocks (vanilla building/natural blocks, Create casings)
  block      64  every other block texture (sources over 64x64 px at 32)
  item       64  items
  animated   32  animation strips (every frame is stored, so they are costly)
"""
import json
import re
import sys

BUDGET = 235e6

HERO_VANILLA = re.compile(
    r"^(stone|smooth_stone(_slab_side)?|cobblestone|mossy_cobblestone|stone_bricks|mossy_stone_bricks|cracked_stone_bricks|"
    r"chiseled_stone_bricks|deepslate(_top)?|cobbled_deepslate|polished_deepslate|deepslate_bricks|deepslate_tiles|"
    r"(polished_)?(andesite|diorite|granite)|tuff(_bricks)?|polished_tuff|calcite|dripstone_block|bricks|mud_bricks|"
    r"(cut_|chiseled_|smooth_)?(red_)?sandstone(_top|_bottom)?|sand|red_sand|gravel|dirt|coarse_dirt|rooted_dirt|"
    r"grass_block_(top|side|side_overlay)|podzol_(top|side)|mud|clay|packed_mud|moss_block|"
    r"\w+_planks|\w+_log(_top)?|stripped_\w+_log(_top)?|\w+_stem(_top)?|stripped_\w+_stem(_top)?|\w+_leaves|"
    r"glass|tinted_glass|\w+_stained_glass|\w+_wool|\w+_concrete|\w+_terracotta|terracotta|\w+_glazed_terracotta|"
    r"(deepslate_)?\w+_ore|nether_\w+_ore|ancient_debris_side|(iron|gold|copper|diamond|emerald|lapis|redstone|coal|netherite)_block|"
    r"(exposed_|weathered_|oxidized_)?(cut_)?copper|bookshelf|crafting_table_(front|side|top)|furnace_(front|side|top)|"
    r"netherrack|nether_bricks|red_nether_bricks|blackstone(_top)?|polished_blackstone(_bricks)?|basalt_(side|top)|"
    r"end_stone(_bricks)?|purpur_(block|pillar(_top)?)|prismarine(_bricks)?|dark_prismarine|quartz_block_(side|top|bottom)|"
    r"obsidian|crying_obsidian|bone_block_(side|top)|hay_block_(side|top)|snow|ice|packed_ice|blue_ice)$")
HERO_CREATE = re.compile(r"^(andesite|brass|copper|railway|shadow_steel|refined_radiance)_casing(_connected)?$|"
                         r"^industrial_iron_block(_top)?$|^(brass|zinc|andesite_alloy)_block$")


def tier_of(t):
    name = t["path"].split("/")[-1][:-4]
    if t["frames"] > 1:
        return "animated"
    if t["kind"] == "item":
        return "item"
    if t["ns"] == "minecraft" and HERO_VANILLA.match(name) and t["w"] <= 16:
        return "hero"
    if t["ns"] == "create" and HERO_CREATE.match(name):
        return "hero"
    return "block"


PX_PER_16 = {"hero": 256, "block": 64, "item": 64, "animated": 32}


def scale_of(t, tier):
    # a texture's own pixels are treated as 16 per block face (vanilla density)
    s = PX_PER_16[tier] / 16
    if tier == "block" and t["w"] * t["h"] > 64 * 64:
        # big connected sheets and model atlases (Vibrant Vaults, Steam 'n' Rails,
        # Rechiseled): half density, or they alone would overflow the atlas
        s /= 2
    return max(1, int(round(s)))


def main(src, out):
    index = json.load(open(f"{src}/index.json"))
    plan, totals, counts = [], {}, {}
    for t in index["textures"]:
        tier = tier_of(t)
        sc = scale_of(t, tier)
        texels = t["w"] * t["h"] * sc * sc
        plan.append(dict(t, tier=tier, scale=sc, texels=texels))
        totals[tier] = totals.get(tier, 0) + texels
        counts[tier] = counts.get(tier, 0) + 1
    total = sum(totals.values())
    for k in sorted(totals, key=totals.get, reverse=True):
        print(f"{k:9s} {counts[k]:6d} textures  {totals[k] / 1e6:7.1f}M texels")
    print(f"total     {len(plan):6d} textures  {total / 1e6:7.1f}M texels  (budget {BUDGET / 1e6:.0f}M, atlas 268M)")
    json.dump(dict(plan=plan, total=total), open(out, "w"))
    if total > BUDGET:
        raise SystemExit("over budget: lower a tier")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
