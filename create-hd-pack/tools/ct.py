"""Connected-texture layouts.

omni_index() is a line-for-line port of AllCTTypes.OMNIDIRECTIONAL.getTextureIndex
from Create 6.0.10 (mc1.21.1-6.0.10 tag). The 8x8 sheet is read row-major:
tile column = index % 8, row = index // 8 (CTSpriteShiftEntry.getTargetU/V).
"""
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Ctx:
    up: bool
    down: bool
    left: bool
    right: bool
    topLeft: bool = False
    topRight: bool = False
    bottomLeft: bool = False
    bottomRight: bool = False

    def inner(self):
        """Inner corners: both sides connected but the diagonal is not."""
        return (self.up and self.left and not self.topLeft,
                self.up and self.right and not self.topRight,
                self.down and self.left and not self.bottomLeft,
                self.down and self.right and not self.bottomRight)

    def shape(self):
        """Everything that changes how a tile looks."""
        return (self.up, self.down, self.left, self.right) + self.inner()


def omni_index(c: Ctx) -> int:
    tileX = tileY = 0
    borders = (not c.up) + (not c.down) + (not c.left) + (not c.right)
    if c.up:
        tileX += 1
    if c.down:
        tileX += 2
    if c.left:
        tileY += 1
    if c.right:
        tileY += 2

    if borders == 0:
        if c.topRight:
            tileX += 1
        if c.topLeft:
            tileX += 2
        if c.bottomRight:
            tileY += 2
        if c.bottomLeft:
            tileY += 1

    if borders == 1:
        if not c.right:
            if c.topLeft or c.bottomLeft:
                tileY = 4
                tileX = -1 + (1 if c.bottomLeft else 0) + (1 if c.topLeft else 0) * 2
        if not c.left:
            if c.topRight or c.bottomRight:
                tileY = 5
                tileX = -1 + (1 if c.bottomRight else 0) + (1 if c.topRight else 0) * 2
        if not c.down:
            if c.topLeft or c.topRight:
                tileY = 6
                tileX = -1 + (1 if c.topLeft else 0) + (1 if c.topRight else 0) * 2
        if not c.up:
            if c.bottomLeft or c.bottomRight:
                tileY = 7
                tileX = -1 + (1 if c.bottomLeft else 0) + (1 if c.bottomRight else 0) * 2

    if borders == 2:
        if ((c.up and c.left and c.topLeft) or (c.down and c.left and c.bottomLeft)
                or (c.up and c.right and c.topRight) or (c.down and c.right and c.bottomRight)):
            tileX += 3

    return tileX + 8 * tileY


def all_contexts():
    """Every context Create can produce (diagonals only count when both sides connect)."""
    seen = set()
    for up, down, left, right, tl, tr, bl, br in product((False, True), repeat=8):
        c = Ctx(up, down, left, right, tl and up and left, tr and up and right,
                bl and down and left, br and down and right)
        if c not in seen:
            seen.add(c)
            yield c


def omni_tiles():
    """index -> Ctx for each of the 47 used tiles; asserts the mapping is unambiguous."""
    tiles = {}
    for c in all_contexts():
        i = omni_index(c)
        assert 0 <= i < 64, (c, i)
        if i in tiles:
            assert tiles[i].shape() == c.shape(), (i, tiles[i], c)
        else:
            tiles[i] = c
    return tiles


# Continuity (OptiFine "method=ctm") 47-tile layout. Copied verbatim from
# Continuity's CtmSpriteProvider.SPRITE_INDEX_MAP (identical on its 1.21/neoforge
# and 1.20.1 branches). Index bits, in texture space:
#   128 64  32
#   1   *   16
#   2   4   8
# Corner bits are only set when both adjacent sides connect.
CTM_SPRITE_INDEX_MAP = [
    0, 3, 0, 3, 12, 5, 12, 15, 0, 3, 0, 3, 12, 5, 12, 15,
    1, 2, 1, 2, 4, 7, 4, 29, 1, 2, 1, 2, 13, 31, 13, 14,
    0, 3, 0, 3, 12, 5, 12, 15, 0, 3, 0, 3, 12, 5, 12, 15,
    1, 2, 1, 2, 4, 7, 4, 29, 1, 2, 1, 2, 13, 31, 13, 14,
    36, 17, 36, 17, 24, 19, 24, 43, 36, 17, 36, 17, 24, 19, 24, 43,
    16, 18, 16, 18, 6, 46, 6, 21, 16, 18, 16, 18, 28, 9, 28, 22,
    36, 17, 36, 17, 24, 19, 24, 43, 36, 17, 36, 17, 24, 19, 24, 43,
    37, 40, 37, 40, 30, 8, 30, 34, 37, 40, 37, 40, 25, 23, 25, 45,
    0, 3, 0, 3, 12, 5, 12, 15, 0, 3, 0, 3, 12, 5, 12, 15,
    1, 2, 1, 2, 4, 7, 4, 29, 1, 2, 1, 2, 13, 31, 13, 14,
    0, 3, 0, 3, 12, 5, 12, 15, 0, 3, 0, 3, 12, 5, 12, 15,
    1, 2, 1, 2, 4, 7, 4, 29, 1, 2, 1, 2, 13, 31, 13, 14,
    36, 39, 36, 39, 24, 41, 24, 27, 36, 39, 36, 39, 24, 41, 24, 27,
    16, 42, 16, 42, 6, 20, 6, 10, 16, 42, 16, 42, 28, 35, 28, 44,
    36, 39, 36, 39, 24, 41, 24, 27, 36, 39, 36, 39, 24, 41, 24, 27,
    37, 38, 37, 38, 30, 11, 30, 32, 37, 38, 37, 38, 25, 33, 25, 26,
]


def ctm_bits(c: Ctx) -> int:
    return ((1 if c.left else 0) | (2 if c.bottomLeft else 0) | (4 if c.down else 0)
            | (8 if c.bottomRight else 0) | (16 if c.right else 0) | (32 if c.topRight else 0)
            | (64 if c.up else 0) | (128 if c.topLeft else 0))


def ctm_index(c: Ctx) -> int:
    return CTM_SPRITE_INDEX_MAP[ctm_bits(c)]


def ctm_tiles():
    """tile -> Ctx for Continuity's 47 tiles; asserts the mapping is unambiguous."""
    tiles = {}
    for c in all_contexts():
        i = ctm_index(c)
        if i in tiles:
            assert tiles[i].shape() == c.shape(), (i, tiles[i], c)
        else:
            tiles[i] = c
    assert sorted(tiles) == list(range(47))
    return tiles


if __name__ == "__main__":
    print(len(ctm_tiles()), "continuity tiles")
    t = omni_tiles()
    print(len(t), "distinct tiles")
    for i in sorted(t):
        c = t[i]
        print(f"{i:2d} ({i % 8},{i // 8})", "".join(k[0] if getattr(c, k) else "." for k in ("up", "down", "left", "right")),
              "inner:", "".join("x" if v else "." for v in c.inner()))
