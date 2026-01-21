# Experiment Analyzer Skill

A Claude Code skill that analyzes LLM experiment results to identify performance issues and recommend improvements.

## Prerequisites

Ensure `ddtool` is installed and authenticated. If not installed, refer to the [ddtool documentation](https://datadoghq.atlassian.net/wiki/spaces/ENG/pages/2396684402/ddtool#Installation).

Verify installation:
```bash
ddtool auth whoami --datacenter us1.staging.dog
```

If not authenticated, log in:
```bash
ddtool auth login --datacenter us1.staging.dog
```

## Installation

### Step 1: Add the MCP Server

```bash
claude mcp add --scope user --transport http "llm-obs-mcp" 'https://llm-obs-mcp.mcp.us1.staging.dog/internal/unstable/llm-obs-mcp/mcp'
```

> **Note:** `--scope user` makes this MCP server available in any Claude Code session. You can optionally scope it to a specific project.

### Step 2: Verify the Connection

1. Start a new Claude Code session:
   ```bash
   claude
   ```

2. List your MCP servers with `/mcp`. You should see `llm-obs-mcp`:
   ```
   Manage MCP servers
   5 servers

     User MCPs (/Users/<username>/.claude.json)
   > llm-obs-mcp · connected
   ```

3. Select it by pressing `Enter` to view details:
   ```
   Llm-obs-mcp MCP Server

   Status: connected
   Auth: authenticated
   URL: https://llm-obs-mcp.mcp.us1.staging.dog/internal/unstable/llm-obs-mcp/mcp
   Capabilities: tools
   Tools: 5 tools

   > 1. View tools
     2. Re-authenticate
     3. Clear authentication
     4. Reconnect
     5. Disable
   ```

### Step 3: Authenticate (if needed)

If not yet authenticated, select **Re-authenticate**. This opens a link to `https://ticino.us1.staging.dog/...` in your browser.

> **Tip:** If authentication fails, try again.

Once authenticated, you can view the available tools via `/mcp` > `llm-obs-mcp` > `View tools`.

### Step 4: Install the Skill

Create the skill directory and copy the skill file:

```bash
mkdir -p ~/.claude/skills/experiment-analyzer
cp path/to/SKILL.md ~/.claude/skills/experiment-analyzer/SKILL.md
```

## Usage

Once installed, claude code will be able to invoke this skill by prompting it. For example:
```
Analyze this experiment <experiment-id>
```

If this is your first time using it, it may ask for permission to invoke the mcp tool calls.
