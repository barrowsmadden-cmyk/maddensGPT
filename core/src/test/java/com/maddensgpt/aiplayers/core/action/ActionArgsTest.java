package com.maddensgpt.aiplayers.core.action;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.maddensgpt.aiplayers.core.ai.ToolSpec;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Optional;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ActionArgsTest {
    @Test
    void readsTypedValues() throws Exception {
        ActionArgs args = ActionArgs.parse("{\"x\": 10, \"y\": 64.0, \"z\": \"-3\", \"count\": 2.6, "
                + "\"item\": \"Oak Log\", \"flag\": \"true\", \"empty\": \"  \"}");
        assertEquals(10, args.getInt("x"));
        assertEquals(64, args.getInt("y"));
        assertEquals(-3, args.getInt("z"), "numeric strings are accepted");
        assertEquals(3, args.getInt("count"), "fractions round to the nearest integer");
        assertEquals("minecraft:oak_log", args.getResourceId("item"));
        assertTrue(args.optBoolean("flag", false));
        assertEquals(Optional.empty(), args.optString("empty"), "blank strings count as missing");
        assertEquals(7, args.optInt("missing", 7));
    }

    @Test
    void explainsBadArguments() throws Exception {
        ActionArgs args = ActionArgs.parse("{\"x\": \"far\", \"n\": 100, \"id\": \"bad id!\", \"o\": {}}");
        assertEquals("'x' must be a number", assertThrows(InvalidArgsException.class, () -> args.getInt("x")).getMessage());
        assertEquals("missing required argument 'y'",
                assertThrows(InvalidArgsException.class, () -> args.getInt("y")).getMessage());
        assertEquals("'n' must be between 1 and 64, got 100",
                assertThrows(InvalidArgsException.class, () -> args.getInt("n", 1, 64)).getMessage());
        assertThrows(InvalidArgsException.class, () -> args.getResourceId("id"));
        assertThrows(InvalidArgsException.class, () -> args.getString("o"));
        assertThrows(InvalidArgsException.class, () -> ActionArgs.parse("[1,2]"));
        assertThrows(InvalidArgsException.class, () -> ActionArgs.parse("{not json"));
    }

    @Test
    void emptyArgumentsAreAnEmptyObject() throws Exception {
        assertFalse(ActionArgs.parse("").has("anything"));
        assertFalse(ActionArgs.parse(null).has("anything"));
    }

    @Test
    void keepsExplicitNamespaces() throws Exception {
        assertEquals("create:brass_ingot", ActionArgs.parse("{\"i\":\"create:brass_ingot\"}").getResourceId("i"));
    }

    @Test
    void catalogSchemasAreConsistent() {
        Set<String> names = new HashSet<>();
        for (ToolSpec spec : ActionCatalog.all()) {
            assertTrue(names.add(spec.name()), "duplicate tool " + spec.name());
            assertTrue(spec.name().matches("[a-z_]+"), spec.name());
            assertFalse(spec.description().isBlank());
            JsonObject params = spec.parameters();
            assertEquals("object", params.get("type").getAsString());
            JsonObject properties = params.getAsJsonObject("properties");
            for (JsonElement required : params.getAsJsonArray("required")) {
                assertTrue(properties.has(required.getAsString()),
                        spec.name() + " requires undeclared " + required.getAsString());
            }
        }
        for (String brain : ActionCatalog.BRAIN_ACTIONS) {
            assertTrue(names.contains(brain), brain);
        }
        assertEquals(ActionCatalog.all().size(), ActionCatalog.BRAIN_ACTIONS.size() + ActionCatalog.gameActions().size());
        for (ToolSpec game : ActionCatalog.gameActions()) {
            assertFalse(ActionCatalog.isBrainAction(game.name()), game.name());
        }
    }
}
