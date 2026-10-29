package com.maddensgpt.aiplayers.core.world;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ObservationTest {
    static Observation sample() {
        return new Observation(
                new Observation.Vec3(12.5, 64, -30.25), "minecraft:overworld", "minecraft:plains", 13_500, "clear",
                18, 20, 15, 3, "minecraft:stone_pickaxe", List.of(),
                List.of(new Observation.ItemCount("minecraft:oak_log", 12), new Observation.ItemCount("minecraft:torch", 4)),
                30, "minecraft:grass_block", "minecraft:oak_log at (13, 65, -30)",
                List.of(new Observation.EntitySighting(123, "minecraft:zombie", null, new Observation.BlockPos(18, 64, -30),
                                6.2, 20, true),
                        new Observation.EntitySighting(45, "minecraft:player", "Steve", new Observation.BlockPos(10, 64, -31),
                                3.0, -1, false)),
                List.of(new Observation.BlockSighting("minecraft:iron_ore", 2, new Observation.BlockPos(5, 40, -20), 26.4)),
                List.of("Steve"));
    }

    @Test
    void rendersCompactReport() {
        String text = sample().render();
        assertTrue(text.contains("Position (12.5, 64.0, -30.3) in overworld, biome plains"), text);
        assertTrue(text.contains("Time: night (13500/24000), weather clear"), text);
        assertTrue(text.contains("Health 18/20, food 15/20, XP level 3"), text);
        assertTrue(text.contains("Holding: stone_pickaxe. Armor: none"), text);
        assertTrue(text.contains("Inventory (30 free slots): oak_log x12, torch x4"), text);
        assertTrue(text.contains("#123 zombie [hostile] 6.2m at (18, 64, -30) hp 20"), text);
        assertTrue(text.contains("#45 player \"Steve\" 3.0m at (10, 64, -31)"), text);
        assertTrue(text.contains("iron_ore x2 (nearest (5, 40, -20), 26.4m)"), text);
        assertTrue(text.contains("Online players: Steve"), text);
    }

    @Test
    void rendersBriefSummary() {
        assertEquals("Position (12.5, 64.0, -30.3) in overworld, night, health 18/20, food 15/20, holding stone_pickaxe",
                sample().renderBrief());
    }

    @Test
    void namesTimesOfDay() {
        assertEquals("morning", Observation.timeOfDay(0));
        assertEquals("afternoon", Observation.timeOfDay(6_000));
        assertEquals("dusk", Observation.timeOfDay(12_500));
        assertEquals("night", Observation.timeOfDay(18_000));
        assertEquals("dawn", Observation.timeOfDay(23_500));
        assertEquals("morning", Observation.timeOfDay(24_000 * 5 + 100), "day time keeps counting across days");
    }
}
