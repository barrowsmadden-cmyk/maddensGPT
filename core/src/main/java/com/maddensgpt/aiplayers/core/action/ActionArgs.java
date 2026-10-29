package com.maddensgpt.aiplayers.core.action;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;

import java.util.Locale;
import java.util.Optional;

/**
 * Typed access to a tool call's JSON arguments. Getters throw {@link InvalidArgsException} with a
 * message written for the model, so it can correct itself on the next turn.
 */
public final class ActionArgs {
    private final JsonObject json;

    private ActionArgs(JsonObject json) {
        this.json = json;
    }

    public static ActionArgs parse(String rawArguments) throws InvalidArgsException {
        if (rawArguments == null || rawArguments.isBlank()) {
            return new ActionArgs(new JsonObject());
        }
        try {
            JsonElement root = JsonParser.parseString(rawArguments);
            if (!root.isJsonObject()) {
                throw new InvalidArgsException("arguments must be a JSON object");
            }
            return new ActionArgs(root.getAsJsonObject());
        } catch (JsonParseException e) {
            throw new InvalidArgsException("arguments are not valid JSON");
        }
    }

    public static ActionArgs of(JsonObject json) {
        return new ActionArgs(json.deepCopy());
    }

    public boolean has(String name) {
        JsonElement value = json.get(name);
        return value != null && !value.isJsonNull();
    }

    public String getString(String name) throws InvalidArgsException {
        return optString(name).orElseThrow(() -> new InvalidArgsException("missing required argument '" + name + "'"));
    }

    public Optional<String> optString(String name) throws InvalidArgsException {
        JsonElement value = json.get(name);
        if (value == null || value.isJsonNull()) {
            return Optional.empty();
        }
        if (!value.isJsonPrimitive()) {
            throw new InvalidArgsException("'" + name + "' must be a string");
        }
        String text = value.getAsString().trim();
        return text.isEmpty() ? Optional.empty() : Optional.of(text);
    }

    public double getDouble(String name) throws InvalidArgsException {
        return number(name).orElseThrow(() -> new InvalidArgsException("missing required argument '" + name + "'"));
    }

    public double optDouble(String name, double fallback) throws InvalidArgsException {
        return number(name).orElse(fallback);
    }

    /** Integer argument. Accepts whole-valued numbers such as 3.0, and rounds others to the nearest integer. */
    public int getInt(String name) throws InvalidArgsException {
        return toInt(name, getDouble(name));
    }

    public int optInt(String name, int fallback) throws InvalidArgsException {
        Optional<Double> value = number(name);
        return value.isPresent() ? toInt(name, value.get()) : fallback;
    }

    /** Integer argument constrained to [min, max]. */
    public int getInt(String name, int min, int max) throws InvalidArgsException {
        return checkRange(name, getInt(name), min, max);
    }

    public int optInt(String name, int fallback, int min, int max) throws InvalidArgsException {
        return checkRange(name, optInt(name, fallback), min, max);
    }

    public boolean optBoolean(String name, boolean fallback) throws InvalidArgsException {
        JsonElement value = json.get(name);
        if (value == null || value.isJsonNull()) {
            return fallback;
        }
        if (value.isJsonPrimitive() && value.getAsJsonPrimitive().isBoolean()) {
            return value.getAsBoolean();
        }
        if (value.isJsonPrimitive()) {
            String text = value.getAsString().trim().toLowerCase(Locale.ROOT);
            if (text.equals("true")) {
                return true;
            }
            if (text.equals("false")) {
                return false;
            }
        }
        throw new InvalidArgsException("'" + name + "' must be true or false");
    }

    /**
     * A Minecraft resource ID such as "oak_log" or "minecraft:oak_log", lower-cased with the
     * "minecraft:" namespace added when none is given.
     */
    public String getResourceId(String name) throws InvalidArgsException {
        return normalizeResourceId(name, getString(name));
    }

    public Optional<String> optResourceId(String name) throws InvalidArgsException {
        Optional<String> raw = optString(name);
        return raw.isPresent() ? Optional.of(normalizeResourceId(name, raw.get())) : Optional.empty();
    }

    static String normalizeResourceId(String name, String raw) throws InvalidArgsException {
        String id = raw.trim().toLowerCase(Locale.ROOT).replace(' ', '_');
        if (!id.contains(":")) {
            id = "minecraft:" + id;
        }
        if (!id.matches("[a-z0-9_.-]+:[a-z0-9_./-]+")) {
            throw new InvalidArgsException("'" + name + "' must be an ID like oak_log or minecraft:oak_log, got '"
                    + raw + "'");
        }
        return id;
    }

    private Optional<Double> number(String name) throws InvalidArgsException {
        JsonElement value = json.get(name);
        if (value == null || value.isJsonNull()) {
            return Optional.empty();
        }
        if (value.isJsonPrimitive()) {
            JsonPrimitive primitive = value.getAsJsonPrimitive();
            try {
                double number = primitive.isNumber() ? primitive.getAsDouble()
                        : Double.parseDouble(primitive.getAsString().trim());
                if (Double.isFinite(number)) {
                    return Optional.of(number);
                }
            } catch (NumberFormatException e) {
                // fall through to the error below
            }
        }
        throw new InvalidArgsException("'" + name + "' must be a number");
    }

    private static int toInt(String name, double value) throws InvalidArgsException {
        if (value > Integer.MAX_VALUE || value < Integer.MIN_VALUE) {
            throw new InvalidArgsException("'" + name + "' is out of range");
        }
        return (int) Math.round(value);
    }

    private static int checkRange(String name, int value, int min, int max) throws InvalidArgsException {
        if (value < min || value > max) {
            throw new InvalidArgsException("'" + name + "' must be between " + min + " and " + max + ", got " + value);
        }
        return value;
    }

    @Override
    public String toString() {
        return json.toString();
    }
}
