"""Original vs remastered item (lit) at a given scale."""
import sys
import numpy as np
from PIL import Image
from remaster import remaster_item
from render import tile_view


def main(src, dest, scale, items):
    cells = []
    for it in items:
        ns, rel = it.split("/", 1)
        maps = remaster_item(f"{src}/{ns}/{rel}", scale, rel)
        maps = [m[:m.shape[1]] for m in maps]
        o = Image.open(f"{src}/{ns}/{rel}").convert("RGBA")
        o = o.crop((0, 0, o.width, o.width)).resize((192, 192), Image.NEAREST)
        bg = Image.new("RGBA", (192, 192), (40, 40, 44, 255)); bg.alpha_composite(o)
        al = Image.new("RGBA", (192, 192), (40, 40, 44, 255)); al.alpha_composite(Image.fromarray(maps[0]).resize((192, 192), Image.LANCZOS))
        lit = tile_view(maps); lb = Image.new("RGBA", lit.size, (40, 40, 44, 255)); lb.alpha_composite(lit)
        cells.append((bg, al, lb.resize((192, 192), Image.LANCZOS)))
    sheet = Image.new("RGB", (len(cells) * 200 + 8, 600), (28, 28, 30))
    for i, cs in enumerate(cells):
        for j, c in enumerate(cs):
            sheet.paste(c.convert("RGB"), (8 + i * 200, 4 + j * 198))
    sheet.save(dest)
    print(dest)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:])
