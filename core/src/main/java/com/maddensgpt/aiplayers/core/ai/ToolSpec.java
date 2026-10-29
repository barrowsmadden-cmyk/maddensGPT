package com.maddensgpt.aiplayers.core.ai;

import com.google.gson.JsonObject;

import java.util.Objects;

/**
 * A function the model may call, described with a JSON Schema for its arguments.
 */
public record ToolSpec(String name, String description, JsonObject parameters) {
    public ToolSpec {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(description, "description");
        Objects.requireNonNull(parameters, "parameters");
    }

    JsonObject toJson() {
        JsonObject function = new JsonObject();
        function.addProperty("name", name);
        function.addProperty("description", description);
        function.add("parameters", parameters.deepCopy());

        JsonObject json = new JsonObject();
        json.addProperty("type", "function");
        json.add("function", function);
        return json;
    }
}
