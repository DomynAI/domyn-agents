---
title: Files, commands and approvals
sidebar_label: Files, commands and approvals
description: "The file, command and browser tools of the CLI agent, the workspace they are confined to, and how you approve what the agent does"
---

# Files, commands and approvals

The agent the setup wizard configures can work in the directory you start the session from: it reads and edits files and runs commands there, and it can browse the web. This page covers what it can reach and how you stay in control of what it changes.

## The tools

The setup wizard asks which sets of tools the agent gets, with all three ticked the first time. Untick one to leave it out:

| Set | Tools | Approval |
|---|---|---|
| `files` | `read_file`, `list_dir`, `glob_files`, `grep` | Never asked: they only read |
| | `write_file`, `append_file`, `edit_file` | Asked for every file |
| `shell` | `run_command` | Asked for every command |
| `browser` | `browser_open`, `browser_snapshot`, `browser_read`, `browser_scroll`, `browser_back` | Never asked: they open and read pages |
| | `browser_click`, `browser_type`, `browser_press` | Asked for every action |

`read_file` also reads documents: PDF, Word, Excel, PowerPoint, OpenDocument, RTF and EPUB files are converted to text on your machine. A PDF is read page by page. A scanned page is read from the text its scanner recognized, when the scanner left it in the file; otherwise it is not read, nor is the content of a large image, and the agent is told which pages and images were left out.

An agent you run from your own file has only the tools you gave it, not these.

A tool result too long to send to the model whole is cut down to its start and its end, and the agent is told how to read the rest with `read_tool_result`. That tool is in no set and is never asked: it only reads results already in the conversation.

### The workspace

The directory you start the session from is the agent's **workspace**. The file tools cannot reach anything outside it, even with your approval, and that includes a symbolic link inside the workspace that points outside. The one exception is the folders of the [skills](../guides/skills.md) the agent loads, so it can read a skill's files and edit a skill when you ask; writing there still needs your approval. Your standing instructions are not part of it when they are outside the workspace. To give the agent a project, `cd` into it before you run `domyn`.

Commands also start in the workspace, but a command can reach any path you could reach yourself. That is why every command asks for approval.

### What commands can see

- **Their own Python environment.** Each conversation gets a virtual environment under `<workspace>/.domyn/conversations/<conversation_id>/venv`, so a package the agent installs stays there and cannot break the environment you run `domyn` from. To let commands use your own environment instead, for example when the agent works on that project, set `isolate_environment: false` (see [Configuring the CLI](configuration.md)).
- **Only safe environment variables.** Commands see a fixed set such as `PATH` and `HOME`, not your API keys or anything loaded from `.env`. To pass a variable through, list its name under `allowed_env`.

### Long-running commands

A command that is still running after its timeout (two minutes unless the agent asks for more) is not killed. It continues in the background, and the agent gets a handle to check on it later or stop it. The agent can also start a command in the background from the outset, such as a server or a file watcher, and gets its handle at once. `/ps` lists these commands and `/stop <handle>` stops one.

A command detached by the shell itself, with `&`, `disown` or `setsid -f`, gets no handle: it does not appear in `/ps` and cannot be stopped with `/stop`. The agent is told so when it runs one. They keep running after the session ends, and the session tells you which ones are still running when you leave.

### Browsing

The `browser` set drives a real browser through [agent-browser](https://github.com/vercel-labs/agent-browser). The standalone build of domyn includes it. Otherwise install it yourself, with `npm install -g agent-browser` or `brew install agent-browser`; the setup wizard warns you when it is not on your `PATH`.

The first time the agent uses a browser tool and no Chrome or Chromium is found, domyn downloads Chrome for Testing (about 200 MB) into `~/.domyn/browsers/`, once, and says so. A Chrome you already have is used as it is. To use another browser, set `browser.executable_path`. On Linux, Chrome also needs system libraries (`libnss3` and a few others, or `agent-browser install --with-deps`).

The agent opens a page, reads its text or the list of elements on it, scrolls and goes back without asking. Clicking, typing and pressing a key wait for your approval, which names the site and the element, such as `https://github.com :: click 'Sign in' [@e5]`. Before an approved action runs, the session checks that the browser is still on that page and that the element still has that name; if anything changed, nothing runs and the agent is told why. When the agent types into a field whose name looks like a password, a token or a card number, the approval shows how many characters it types, not the text.

What the browser may open:

- Only `http` and `https` addresses: no local files.
- No address on your machine or your local network, such as `localhost:3000`, unless you set `allow_private_urls` (see [Configuring the CLI](configuration.md)). Cloud metadata addresses are refused even then.
- Any public site, unless you list the only ones allowed in `allowed_domains`.

The text of a page reaches the agent marked as data from the page, not instructions. That makes a page that tries to give the agent orders easier to spot, but it is not a guarantee: the approvals above are what stop an action you did not want. Each conversation has its own browser, closed when the session ends, or after five minutes without a browser command.

## Approvals

When the agent wants to write a file or run a command, the session shows the call in full and asks `run it?`. The heading is what an approval applies to, such as `$ git status` for a command or `write_file path=notes.md` for a file; below it come the arguments it does not already show, exactly as the agent wrote them: the content of a file to write, or an edit as a diff of the lines it removes (`-`) and adds (`+`). A command on several lines is shown as written, under the name of the tool.

The question comes with a numbered list of answers: choose one with the arrow keys, or type its number, and press Enter. If the agent is still working, the list appears in the input line. Choose **no**, or press Ctrl-C, and the agent is told the call was refused, then continues without it. The `yes` answers differ in how long the approval is remembered:

| Answer | Lasts |
|---|---|
| yes, just this once | This call only |
| yes, this exact call, for the rest of this conversation | Until the conversation ends |
| yes, this exact call, in this project | For every session started from this directory |
| yes, this exact call, always | For every session |
| yes, anything matching a pattern, for the rest of this conversation | For the rest of this conversation |

A session started from your home directory offers no project choice, because your home directory is not a project.

The pattern choice appears for commands and browser actions. For a command it proposes a pattern such as `$ git *`, which you can edit before saving; for a browser action, the whole site, such as `https://github.com :: *`, which covers every click, typing and key press on that site. A command made of several parts, such as `git pull && make`, is covered only when every part matches an approved pattern, so approving `$ git *` never approves `git status && rm -rf build`. Commands that run other commands, such as `bash`, `sudo` or `xargs`, never get a pattern proposed.

Approvals that last beyond the conversation are saved in your home directory, never in the project: `~/.domyn/approvals.yaml` for "always", and `~/.domyn/projects/` for a project. `/approvals` lists and removes them.

`/auto`, or Shift+Tab, skips every approval until you turn it off the same way or the session ends. Shift+Tab works at the prompt, while the agent is working, and on an open approval question. Use it only when you trust what the agent is about to do. A short list of commands is refused even then, such as deleting your home or root directory, formatting a disk, or piping a downloaded script into a shell. That list catches accidents; it is not a security boundary.

For approvals in your own agents, see [Tool approvals](../guides/approvals.md).

## Next steps

- [Configuring the CLI](configuration.md): change the tools, the workspace, what commands may see and where the browser may go.
- [Sessions and commands](sessions.md): `/approvals`, `/auto`, `/ps` and the other session commands.
