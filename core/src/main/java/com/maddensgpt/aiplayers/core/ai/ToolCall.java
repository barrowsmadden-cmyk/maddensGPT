package com.maddensgpt.aiplayers.core.ai;

import com.google.gson.JsonObject;

import java.util.Objects;

/**
 * One function call requested by the model.
 *
 * @param id        provider-assigned ID that the matching tool result must echo
 * @param name      function name
 * @param arguments raw JSON arguments string exactly as the model produced it (may be malformed)
 */
public record ToolCall(String id, String name, String arguments) {
    public ToolCall {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(name, "name");
        arguments = arguments == null ? "" : arguments;
    }

    JsonObject toJson() {
        JsonObject function = new JsonObject();
        function.addProperty("name", name);
        function.addProperty("arguments", arguments);

        JsonObject json = new JsonObject();
        json.addProperty("id", id);
        json.addProperty("type", "function");
        json.add("function", function);
        return json;
    }
}
