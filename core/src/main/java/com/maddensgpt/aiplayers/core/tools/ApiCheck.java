package com.maddensgpt.aiplayers.core.tools;

import com.maddensgpt.aiplayers.core.action.ActionArgs;
import com.maddensgpt.aiplayers.core.action.ActionResult;
import com.maddensgpt.aiplayers.core.action.RunningAction;
import com.maddensgpt.aiplayers.core.ai.OpenAiCompatibleClient;
import com.maddensgpt.aiplayers.core.ai.ProviderConfig;
import com.maddensgpt.aiplayers.core.brain.BotBrain;
import com.maddensgpt.aiplayers.core.brain.BotEnvironment;
import com.maddensgpt.aiplayers.core.brain.BotSettings;
import com.maddensgpt.aiplayers.core.brain.SettingsFile;
import com.maddensgpt.aiplayers.core.world.Observation;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Checks that the configured provider works, without Minecraft: lists the available models, then
 * runs one bot turn against a made-up world and prints what the bot says and tries to do.
 *
 * <p>Usage: {@code ./gradlew :core:checkApi --args="path/to/aiplayers.json"}
 */
public final class ApiCheck {
    private ApiCheck() {
    }

    public static void main(String[] args) throws Exception {
        Path path = Path.of(args.length > 0 ? args[0] : "aiplayers.json");
        BotSettings settings = SettingsFile.loadOrCreate(path);
        ProviderConfig provider = settings.provider;
        System.out.println("Config: " + path.toAbsolutePath());
        System.out.println("Provider: " + provider.provider + " at " + provider.baseUrl + ", model '" + provider.model + "'");

        List<String> problems = provider.validate();
        if (!problems.isEmpty()) {
            System.out.println("\nFix these in the config first:");
            problems.forEach(problem -> System.out.println("  - " + problem));
            System.exit(1);
        }

        try (OpenAiCompatibleClient client = new OpenAiCompatibleClient(provider)) {
            try {
                List<String> models = client.listModels().get(60, TimeUnit.SECONDS);
                System.out.println("\nModels available to this key: " + String.join(", ", models));
                if (!models.isEmpty() && !models.contains(provider.model.trim())) {
                    System.out.println("WARNING: '" + provider.model + "' is not in that list.");
                }
            } catch (Exception e) {
                System.out.println("\nCould not list models (some providers do not support it): " + e.getMessage());
            }

            BotEnvironment env = new BotEnvironment() {
                @Override
                public Observation observe() {
                    return sampleWorld();
                }

                @Override
                public void sendChat(String line) {
                    System.out.println("  <Robo> " + line);
                }

                @Override
                public void log(String message) {
                    System.out.println("  [log] " + message);
                }

                @Override
                public void warn(String message) {
                    System.out.println("  [warn] " + message);
                }
            };
            // Only one request: the gap before a second one is longer than this check runs.
            settings.minSecondsBetweenRequests = 1_000_000;
            BotBrain brain = new BotBrain("Robo", settings, client, ApiCheck::pretend, env);
            brain.onChat("Steve", "hey Robo, can you help me get some wood?", false);

            System.out.println("\nRunning one turn (game actions are only printed, not performed):");
            long deadline = System.currentTimeMillis() + 180_000;
            long idleSince = -1;
            while (System.currentTimeMillis() < deadline) {
                long now = System.currentTimeMillis();
                brain.tick(now);
                if (!brain.status().lastError().isEmpty()) {
                    System.out.println("\nRequest failed: " + brain.status().lastError());
                    System.exit(1);
                }
                if (brain.status().totalRequests() > 0 && brain.state() == BotBrain.State.IDLE) {
                    if (idleSince < 0) {
                        idleSince = now;
                    } else if (now - idleSince > 10_000) {
                        break; // turn finished and queued chat has had time to go out
                    }
                }
                Thread.sleep(50);
            }
            brain.shutdown();
            BotBrain.Status status = brain.status();
            System.out.println("\nDone. Requests: " + status.totalRequests() + ", tokens in/out: "
                    + status.promptTokens() + "/" + status.completionTokens());
        }
    }

    private static RunningAction pretend(String name, ActionArgs args) {
        System.out.println("  -> " + name + " " + args);
        return new RunningAction() {
            @Override
            public ActionResult tick() {
                return ActionResult.ok("(test run: action not performed)");
            }

            @Override
            public void cancel() {
            }
        };
    }

    private static Observation sampleWorld() {
        return new Observation(
                new Observation.Vec3(12.5, 64, -30.5), "minecraft:overworld", "minecraft:forest", 2_000, "clear",
                20, 20, 18, 0, "", List.of(), List.of(), 36, "minecraft:grass_block", "",
                List.of(new Observation.EntitySighting(45, "minecraft:player", "Steve",
                        new Observation.BlockPos(10, 64, -31), 2.5, -1, false)),
                List.of(new Observation.BlockSighting("minecraft:oak_log", 23, new Observation.BlockPos(15, 64, -28), 3.6)),
                List.of("Steve"));
    }
}
