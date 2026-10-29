package com.maddensgpt.aiplayers.core.brain;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Reads and writes {@link BotSettings} as pretty-printed JSON.
 */
public final class SettingsFile {
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create();

    private SettingsFile() {
    }

    /** Loads the file, or writes and returns the defaults when it does not exist yet. */
    public static BotSettings loadOrCreate(Path path) throws IOException {
        if (!Files.exists(path)) {
            BotSettings defaults = new BotSettings();
            save(path, defaults);
            return defaults;
        }
        try (Reader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            BotSettings settings = GSON.fromJson(reader, BotSettings.class);
            if (settings == null) {
                throw new IOException(path + " is empty");
            }
            return settings.sanitize();
        } catch (JsonParseException e) {
            throw new IOException("Could not read " + path + ": " + e.getMessage(), e);
        }
    }

    public static void save(Path path, BotSettings settings) throws IOException {
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (Writer writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            GSON.toJson(settings, writer);
        }
    }
}
