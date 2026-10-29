package com.maddensgpt.aiplayers.core.action;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.maddensgpt.aiplayers.core.ai.ToolSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Every function the model can call. "Brain" actions (talking, notes, waiting) are handled by
 * {@code BotBrain} itself; "game" actions are carried out in the world by an {@link ActionHandler}.
 */
public final class ActionCatalog {
    // Brain actions
    public static final String SAY = "say";
    public static final String REMEMBER = "remember";
    public static final String SET_PLAN = "set_plan";
    public static final String WAIT = "wait";

    // Web actions (brain-handled, offered only when internet access is set up)
    public static final String WEB_SEARCH = "web_search";
    public static final String READ_WEBPAGE = "read_webpage";

    // Game actions
    public static final String MOVE_TO = "move_to";
    public static final String FOLLOW_PLAYER = "follow_player";
    public static final String MINE_BLOCK = "mine_block";
    public static final String COLLECT_BLOCKS = "collect_blocks";
    public static final String PLACE_BLOCK = "place_block";
    public static final String USE_BLOCK = "use_block";
    public static final String ATTACK = "attack";
    public static final String EAT = "eat";
    public static final String EQUIP = "equip";
    public static final String CRAFT = "craft";
    public static final String SMELT = "smelt";
    public static final String PICK_UP_ITEMS = "pick_up_items";
    public static final String GIVE_ITEM = "give_item";
    public static final String DROP_ITEM = "drop_item";
    public static final String STORE_ITEMS = "store_items";
    public static final String TAKE_ITEMS = "take_items";
    public static final String FIND = "find";

    public static final Set<String> BRAIN_ACTIONS = Set.of(SAY, REMEMBER, SET_PLAN, WAIT, WEB_SEARCH, READ_WEBPAGE);

    private static final List<ToolSpec> BRAIN_SPECS = List.of(
            tool(SAY, "Send a message in the in-game chat. This is the ONLY way other players see what you say "
                            + "or think. Keep it short and natural, like a real player typing.",
                    new Schema().string("message", "What to say, plain text, under 200 characters.", true)),
            tool(REMEMBER, "Save a fact to your long-term notes, which you see every turn (e.g. base location, "
                            + "where you found iron, what someone asked you to do).",
                    new Schema().string("note", "The fact to keep, one short sentence.", true)),
            tool(SET_PLAN, "Replace your current plan, which you see every turn. Use a few short steps.",
                    new Schema().string("plan", "The plan, e.g. '1. get 8 logs 2. make planks and a table "
                            + "3. craft a wooden pickaxe'. Empty to clear.", true)),
            tool(WAIT, "Do nothing for a while, e.g. to wait for someone or for night to pass.",
                    new Schema().integer("seconds", "How long to wait, 1 to 60.", true)));

    private static final ToolSpec WEB_SEARCH_SPEC = tool(WEB_SEARCH,
            "Search the internet. Use it for things you are unsure about (recipes, mob behaviour, mod items, "
                    + "version changes) or when a player asks a question you cannot answer from the game. "
                    + "Returns titles, links and short snippets.",
            new Schema().string("query", "What to search for, e.g. 'minecraft how to make a brewing stand'.", true));

    private static final ToolSpec READ_WEBPAGE_SPEC = tool(READ_WEBPAGE,
            "Download a public web page and read its text, e.g. a link from web_search results.",
            new Schema()
                    .string("url", "The page address, starting with https://.", true)
                    .integer("start", "Character offset to continue reading a long page from. Default 0.", false));

    private static final List<ToolSpec> GAME_SPECS = List.of(
            tool(MOVE_TO, "Walk to a block position using pathfinding. Fails if there is no path.",
                    position(new Schema(), "Target")
                            .number("range", "Stop once within this many blocks. Default 1.", false)),
            tool(FOLLOW_PLAYER, "Follow a player around for a while, staying close to them.",
                    new Schema()
                            .string("player", "Name of the player to follow.", true)
                            .number("distance", "How close to stay, in blocks. Default 3.", false)
                            .integer("seconds", "How long to follow, 1 to 120. Default 30.", false)),
            tool(MINE_BLOCK, "Walk within reach of one block and break it with the best tool you have, then pick up "
                            + "what drops.",
                    position(new Schema(), "Block")),
            tool(COLLECT_BLOCKS, "Find and mine several blocks of one type nearby, picking up the drops. Best way to "
                            + "gather wood, stone, ores, sand, etc.",
                    new Schema()
                            .string("block", "Block ID, e.g. oak_log, stone, iron_ore.", true)
                            .integer("count", "How many blocks to mine, 1 to 64.", true)
                            .integer("radius", "Search radius in blocks, 4 to 48. Default 32.", false)),
            tool(PLACE_BLOCK, "Place a block from your inventory at a position next to an existing block.",
                    position(new Schema().string("item", "Block item ID to place, e.g. cobblestone, torch.", true),
                            "Target")),
            tool(USE_BLOCK, "Right-click a block: open a door, press a button, pull a lever, sleep in a bed, etc.",
                    position(new Schema(), "Block")),
            tool(ATTACK, "Attack a mob or entity until it dies or gets away. Equip a weapon first if you have one.",
                    new Schema()
                            .integer("entity_id", "The #id of the entity from your status report.", false)
                            .string("entity_type", "Or: attack the nearest entity of this type, e.g. zombie, cow.",
                                    false)),
            tool(EAT, "Eat food from your inventory until full or out of that food.",
                    new Schema().string("item", "Food item ID. Leave out to pick the best food you have.", false)),
            tool(EQUIP, "Equip an item from your inventory into your hand or an armor slot.",
                    new Schema()
                            .string("item", "Item ID, e.g. iron_sword, iron_helmet.", true)
                            .enumeration("slot", "Where to put it. Default: mainhand, or the matching armor slot.",
                                    false, "mainhand", "offhand", "head", "chest", "legs", "feet")),
            tool(CRAFT, "Craft items from your inventory. Uses a crafting table within 4 blocks for 3x3 recipes.",
                    new Schema()
                            .string("item", "Item ID to craft, e.g. oak_planks, crafting_table, stone_pickaxe.", true)
                            .integer("count", "How many items you want, 1 to 64. Default 1.", false)),
            tool(SMELT, "Smelt or cook items in a furnace within 4 blocks, waiting for the result.",
                    new Schema()
                            .string("item", "Item ID to put in, e.g. raw_iron, beef.", true)
                            .integer("count", "How many, 1 to 64. Default 1.", false)
                            .string("fuel", "Fuel item ID. Leave out to use coal, charcoal or wood you have.", false)),
            tool(PICK_UP_ITEMS, "Walk over dropped items nearby to pick them up.",
                    new Schema().integer("radius", "Search radius in blocks, 1 to 16. Default 8.", false)),
            tool(GIVE_ITEM, "Walk to a player and toss them items.",
                    new Schema()
                            .string("player", "Player name.", true)
                            .string("item", "Item ID.", true)
                            .integer("count", "How many, 1 to 64. Default 1.", false)),
            tool(DROP_ITEM, "Drop items on the ground where you stand.",
                    new Schema()
                            .string("item", "Item ID.", true)
                            .integer("count", "How many, 1 to 64. Default: all of that item.", false)),
            tool(STORE_ITEMS, "Put items into a chest or barrel.",
                    position(new Schema()
                            .string("item", "Item ID to store. Leave out to store everything except tools, "
                                    + "armor and food.", false)
                            .integer("count", "How many, 1 to 64. Default: all of that item.", false), "Container")),
            tool(TAKE_ITEMS, "Take items out of a chest or barrel.",
                    position(new Schema()
                            .string("item", "Item ID to take.", true)
                            .integer("count", "How many, 1 to 64. Default: all of that item.", false), "Container")),
            tool(FIND, "Search around you for a block or entity type and get the closest positions. Takes no time.",
                    new Schema()
                            .string("target", "Block or entity ID, e.g. iron_ore, crafting_table, cow.", true)
                            .integer("radius", "Search radius in blocks, 4 to 64. Default 32.", false)));

    private static final List<ToolSpec> ALL_SPECS;

    static {
        List<ToolSpec> all = new ArrayList<>(BRAIN_SPECS);
        all.add(WEB_SEARCH_SPEC);
        all.add(READ_WEBPAGE_SPEC);
        all.addAll(GAME_SPECS);
        ALL_SPECS = List.copyOf(all);
    }

    private ActionCatalog() {
    }

    /** Every function, including the web ones, in the order they are offered to the model. */
    public static List<ToolSpec> all() {
        return ALL_SPECS;
    }

    /** The functions to offer, leaving out web access the server has not set up. */
    public static List<ToolSpec> offered(boolean webSearch, boolean readPages) {
        return ALL_SPECS.stream()
                .filter(spec -> webSearch || !spec.name().equals(WEB_SEARCH))
                .filter(spec -> readPages || !spec.name().equals(READ_WEBPAGE))
                .toList();
    }

    public static List<ToolSpec> gameActions() {
        return GAME_SPECS;
    }

    public static boolean isBrainAction(String name) {
        return BRAIN_ACTIONS.contains(name);
    }

    public static boolean isKnown(String name) {
        return ALL_SPECS.stream().anyMatch(spec -> spec.name().equals(name));
    }

    private static ToolSpec tool(String name, String description, Schema parameters) {
        return new ToolSpec(name, description, parameters.build());
    }

    private static Schema position(Schema schema, String what) {
        return schema
                .integer("x", what + " block X.", true)
                .integer("y", what + " block Y (height).", true)
                .integer("z", what + " block Z.", true);
    }

    /** Minimal builder for the JSON Schema "object" used as function parameters. */
    private static final class Schema {
        private final JsonObject properties = new JsonObject();
        private final JsonArray required = new JsonArray();

        Schema string(String name, String description, boolean isRequired) {
            return property(name, "string", description, isRequired);
        }

        Schema integer(String name, String description, boolean isRequired) {
            return property(name, "integer", description, isRequired);
        }

        Schema number(String name, String description, boolean isRequired) {
            return property(name, "number", description, isRequired);
        }

        Schema enumeration(String name, String description, boolean isRequired, String... values) {
            property(name, "string", description, isRequired);
            JsonArray options = new JsonArray();
            for (String value : values) {
                options.add(value);
            }
            properties.getAsJsonObject(name).add("enum", options);
            return this;
        }

        private Schema property(String name, String type, String description, boolean isRequired) {
            JsonObject property = new JsonObject();
            property.addProperty("type", type);
            property.addProperty("description", description);
            properties.add(name, property);
            if (isRequired) {
                required.add(name);
            }
            return this;
        }

        JsonObject build() {
            JsonObject schema = new JsonObject();
            schema.addProperty("type", "object");
            schema.add("properties", properties.deepCopy());
            schema.add("required", required.deepCopy());
            return schema;
        }
    }
}
