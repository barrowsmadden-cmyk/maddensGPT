package com.maddensgpt.aiplayers.core.world;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Snapshot of what the bot can perceive, filled in by the Minecraft side each turn and rendered
 * as text for the model. Item, block and entity names are resource IDs such as "minecraft:oak_log";
 * the "minecraft:" prefix is dropped when rendering to save tokens.
 */
public record Observation(
        Vec3 position,
        String dimension,
        String biome,
        long dayTime,
        String weather,
        float health,
        float maxHealth,
        int food,
        int xpLevel,
        String heldItem,
        List<String> armor,
        List<ItemCount> inventory,
        int freeInventorySlots,
        String standingOn,
        String lookingAt,
        List<EntitySighting> nearbyEntities,
        List<BlockSighting> notableBlocks,
        List<String> onlinePlayers) {

    public record Vec3(double x, double y, double z) {
        String render() {
            return String.format(Locale.ROOT, "(%.1f, %.1f, %.1f)", x, y, z);
        }
    }

    public record BlockPos(int x, int y, int z) {
        String render() {
            return "(" + x + ", " + y + ", " + z + ")";
        }
    }

    public record ItemCount(String item, int count) {
    }

    /**
     * @param id       entity network ID the model can pass to attack
     * @param type     entity type ID, e.g. "minecraft:zombie"
     * @param name     custom or player name, or null
     * @param health   current health, or a negative value when not applicable
     * @param hostile  whether the mob attacks players on sight
     */
    public record EntitySighting(int id, String type, String name, BlockPos position, double distance,
                                 float health, boolean hostile) {
    }

    /** A kind of block worth mentioning (ores, logs, crafting stations, water, lava...). */
    public record BlockSighting(String block, int count, BlockPos nearest, double distance) {
    }

    public Observation {
        Objects.requireNonNull(position, "position");
        dimension = orEmpty(dimension);
        biome = orEmpty(biome);
        weather = orEmpty(weather);
        heldItem = orEmpty(heldItem);
        standingOn = orEmpty(standingOn);
        lookingAt = orEmpty(lookingAt);
        armor = armor == null ? List.of() : List.copyOf(armor);
        inventory = inventory == null ? List.of() : List.copyOf(inventory);
        nearbyEntities = nearbyEntities == null ? List.of() : List.copyOf(nearbyEntities);
        notableBlocks = notableBlocks == null ? List.of() : List.copyOf(notableBlocks);
        onlinePlayers = onlinePlayers == null ? List.of() : List.copyOf(onlinePlayers);
    }

    /** Full report for the current turn. */
    public String render() {
        StringBuilder out = new StringBuilder();
        out.append("Position ").append(position.render())
                .append(" in ").append(shortId(dimension))
                .append(biome.isEmpty() ? "" : ", biome " + shortId(biome)).append('\n');
        out.append("Time: ").append(timeOfDay(dayTime)).append(" (").append(Math.floorMod(dayTime, 24_000L))
                .append("/24000)").append(weather.isEmpty() ? "" : ", weather " + weather).append('\n');
        out.append(String.format(Locale.ROOT, "Health %.0f/%.0f, food %d/20, XP level %d%n",
                health, maxHealth, food, xpLevel));
        if (!standingOn.isEmpty() || !lookingAt.isEmpty()) {
            out.append("Standing on ").append(standingOn.isEmpty() ? "nothing" : shortId(standingOn));
            if (!lookingAt.isEmpty()) {
                out.append(", looking at ").append(lookingAt);
            }
            out.append('\n');
        }
        out.append("Holding: ").append(heldItem.isEmpty() ? "nothing" : shortId(heldItem))
                .append(". Armor: ").append(armor.isEmpty() ? "none" : joinIds(armor)).append('\n');

        out.append("Inventory (").append(freeInventorySlots).append(" free slots): ");
        if (inventory.isEmpty()) {
            out.append("empty");
        } else {
            List<String> items = new ArrayList<>();
            for (ItemCount item : inventory) {
                items.add(shortId(item.item()) + " x" + item.count());
            }
            out.append(String.join(", ", items));
        }
        out.append('\n');

        out.append("Nearby entities: ");
        if (nearbyEntities.isEmpty()) {
            out.append("none");
        } else {
            List<String> entities = new ArrayList<>();
            for (EntitySighting entity : nearbyEntities) {
                StringBuilder line = new StringBuilder("#").append(entity.id()).append(' ').append(shortId(entity.type()));
                if (entity.name() != null && !entity.name().isEmpty()) {
                    line.append(" \"").append(entity.name()).append('"');
                }
                if (entity.hostile()) {
                    line.append(" [hostile]");
                }
                line.append(String.format(Locale.ROOT, " %.1fm at ", entity.distance())).append(entity.position().render());
                if (entity.health() >= 0) {
                    line.append(String.format(Locale.ROOT, " hp %.0f", entity.health()));
                }
                entities.add(line.toString());
            }
            out.append(String.join("; ", entities));
        }
        out.append('\n');

        out.append("Notable blocks nearby: ");
        if (notableBlocks.isEmpty()) {
            out.append("none");
        } else {
            List<String> blocks = new ArrayList<>();
            for (BlockSighting block : notableBlocks) {
                blocks.add(String.format(Locale.ROOT, "%s x%d (nearest %s, %.1fm)",
                        shortId(block.block()), block.count(), block.nearest().render(), block.distance()));
            }
            out.append(String.join("; ", blocks));
        }
        out.append('\n');

        out.append("Online players: ").append(onlinePlayers.isEmpty() ? "none" : String.join(", ", onlinePlayers));
        return out.toString();
    }

    /** One-line summary used for older turns in the conversation history. */
    public String renderBrief() {
        return String.format(Locale.ROOT, "Position %s in %s, %s, health %.0f/%.0f, food %d/20, holding %s",
                position.render(), shortId(dimension), timeOfDay(dayTime), health, maxHealth, food,
                heldItem.isEmpty() ? "nothing" : shortId(heldItem));
    }

    /** Minecraft day time: 0 is sunrise, 6000 noon, 12000 sunset, 18000 midnight. */
    static String timeOfDay(long dayTime) {
        long t = Math.floorMod(dayTime, 24_000L);
        if (t < 6_000) {
            return "morning";
        }
        if (t < 12_000) {
            return "afternoon";
        }
        if (t < 13_000) {
            return "dusk";
        }
        if (t < 23_000) {
            return "night";
        }
        return "dawn";
    }

    static String shortId(String id) {
        return id.startsWith("minecraft:") ? id.substring("minecraft:".length()) : id;
    }

    private static String joinIds(List<String> ids) {
        List<String> shortened = new ArrayList<>();
        for (String id : ids) {
            shortened.add(shortId(id));
        }
        return String.join(", ", shortened);
    }

    private static String orEmpty(String value) {
        return value == null ? "" : value;
    }
}
