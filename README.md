# maddensGPT: AI Players

A NeoForge 1.21.x mod (in progress) that spawns server-side fake players controlled by an AI model
(OpenAI or DeepSeek). The bots play the game through function calls and talk in chat about what
they are doing and thinking.

## Status

| Part | State |
| --- | --- |
| `core`: AI client, actions, chat narration, think-act loop | Done, unit tested |
| `neoforge`: fake player, carrying out actions, `/aibot` commands | Not started: the build needs `maven.neoforged.net` and the Mojang hosts, which this environment blocks |

## How a bot thinks

1. Each turn the bot sends the model a status report: position, health, food, inventory, nearby
   entities and blocks, recent events, chat since its last turn, and its goal, plan and notes.
2. The model replies with function calls (`say`, `collect_blocks`, `craft`, `attack`, ...). They run
   in order, one at a time; if one fails, the rest are skipped.
3. Every result goes back to the model on the next turn.

Talking goes through the `say` function. The system prompt tells the bot to think out loud like a
person playing with friends ("gonna grab some wood before it gets dark", "ugh creeper, backing off").
Messages are cleaned up, split at sentence boundaries and sent with a short typing delay, so the bot
never floods chat. Any text the model writes outside function calls stays private and goes to the
server log.

Mentioning the bot by name, messaging it privately, hurting it, or changing its goal interrupts what
it is doing so it can respond.

## Configuration (`aiplayers.json`)

The file is created with defaults on first run. Key settings:

- `provider.baseUrl` / `provider.model`: DeepSeek defaults to `https://api.deepseek.com` and
  `deepseek-v4-flash`. For OpenAI, set `baseUrl` to `https://api.openai.com/v1`,
  `maxTokensField` to `max_completion_tokens`, and `model` to a current model ID. Model IDs change
  often, so check your provider's docs.
- `provider.apiKeyEnv`: name of the environment variable that holds your key (recommended), or put
  the key in `provider.apiKey`.
- `provider.temperature`: optional; leave it out to use the model's default.
- `provider.extraBody`: extra request fields, e.g. `{"thinking": {"type": "enabled"}}` for DeepSeek
  thinking mode.
- `chattiness`: `quiet`, `normal` or `chatty`.
- `personality`, `trustedPlayers` (empty means anyone), `extraInstructions`.
- Cost limits: `maxRequestsPerHour`, `minSecondsBetweenRequests`, `idleThinkSeconds`,
  `maxHistoryTurns`.

## Check your API key without Minecraft

```
export DEEPSEEK_API_KEY=sk-...        # or OPENAI_API_KEY
./gradlew :core:checkApi --args="aiplayers.json"
```

This lists the models your key can use and runs one sample turn against a made-up world, printing
what the bot says and which actions it tries. It makes one chat request and one model-list request.

## Build and test

```
./gradlew :core:build
```

Requires Java 21.
