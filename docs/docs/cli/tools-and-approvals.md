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
| `browser` | `browser_snapshot`, `browser_read`, `browser_scroll`, `browser_back` | Never asked: they read the page that is open |
| | `browser_open` | Asked for every page |
| | `browser_click`, `browser_type`, `browser_press` | Asked for every action |

`read_file` also reads documents: PDF, Word, Excel, PowerPoint, OpenDocument, RTF and EPUB files are converted to text on your machine. A PDF is read page by page. A scanned page is read from the text its scanner recognized, when the scanner left it in the file; otherwise it is not read, nor is the content of a large image, and the agent is told which pages and images were left out.

An agent you run from your own file has only the tools you gave it, not these.

A tool result too long to send to the model whole is cut down to its start and its end, and the agent is told how to read the rest with `read_tool_result`. That tool is in no set and is never asked: it only reads results already in the conversation.

### The workspace

The directory you start the session from is the agent's **workspace**. The file tools cannot reach anything outside it, even with your approval, and that includes a symbolic link inside the workspace that points outside. The one exception is the folders of the [skills](../guides/skills.md) the agent loads, so it can read a skill's files and edit a skill when you ask; writing there still needs your approval. Your standing instructions are not part of it when they are outside the workspace. To give the agent a project, `cd` into it before you run `domyn`.

Commands also start in the workspace, but a command can reach any path you could reach yourself. That is why every command asks for approval.

### What commands can see

- **Their own Python environment.** Each conversation gets a virtual environment under `conversations/<conversation_id>/venv` next to your conversation history: in the project's folder when you run one with `--project`, otherwise in `~/.domyn`. So a package the agent installs stays there and cannot break the environment you run `domyn` from. To let commands use your own environment instead, for example when the agent works on that project, set `isolate_environment: false` (see [Configuring the CLI](configuration.md)).
- **Only safe environment variables.** Commands see a fixed set such as `PATH` and `HOME`, not your API keys or anything loaded from `.env`. To pass a variable through, list its name under `allowed_env`.

### Long-running commands

A command that is still running after its timeout (two minutes unless the agent asks for more) is not killed. It continues in the background, and the agent gets a handle to check on it later or stop it. The agent can also start a command in the background from the outset, such as a server or a file watcher, and gets its handle at once. `/ps` lists these commands and `/stop <handle>` stops one.

A command detached by the shell itself, with `&`, `disown` or `setsid -f`, gets no handle: it does not appear in `/ps` and cannot be stopped with `/stop`. The agent is told so when it runs one. They keep running after the session ends, and the session tells you which ones are still running when you leave.

### Running the tools in a Docker container

By default (`backend: auto`), the agent still runs in your terminal, and so do its approvals. What the agent's tools do moves into a Docker container that can write nothing of yours except the workspace: the commands it runs, its file tools and its browser. A file a command downloads is a file `read_file` reads, and a path a skill names means the same thing to all of them. This needs Docker (or Podman, with `docker_binary`) running on your machine.

If Docker is not installed or not running, or its image cannot be pulled, `domyn run` still starts, warns you, and runs everything on your machine instead. The line under the banner says which one you got (`tools: docker` or `tools: this machine`). Start Docker and run `domyn run` again to use the container. Three values of `backend` say what you want:

| Value | Meaning |
|---|---|
| `auto` | A container when Docker works, this machine, with a warning, when it does not |
| `docker` | Always a container: `domyn run` does not start without Docker |
| `local` | Always this machine, with no warning |

If Docker stops during a session, a command fails with a message; the session does not switch to your machine half way, because the files and packages of the conversation are in the container. Restart `domyn run` to choose again.

```yaml
tool_bundles:
- name: cli_tools
  bundles:
  - files
  - shell
  - browser
  backend: auto
  docker:
    allowed_network: public   # the default: any public site over HTTPS
```

:::warning The container is not your machine
Everything the agent's tools do happens in the container, and it differs from running on your machine in ways that matter. If something works for the agent on `backend: local` and not on `backend: docker`, check this list first:

- **`config.yaml` is read-only.** The agent can read it and cannot change it. It names code your machine runs (tools, hooks, MCP servers) and which backend to use, so a command in the container must not be able to rewrite it. To change it, edit it yourself, or run `domyn setup` or `/model`. This also means the agent cannot configure itself from the container, as it can with `backend: local`.
- **Credentials, approvals and history are not there.** Your API keys, the permissions you granted and past conversations are not visible, so a command cannot read them.
- **Only what is mounted exists.** Files outside your workspace, the skill folders and the session's state (`~/Downloads`, `~/.ssh`, other repositories) are not in the container. Copy a file into the workspace for the agent to use it.
- **`.git/hooks` and `.git/config` are read-only.** A hook a command planted would run on your machine at your next `git commit`, and so would a `core.fsmonitor` or `core.hooksPath` entry in the config.
- **Your machine's programs and servers are out of reach.** `localhost` is the container, and addresses on your machine or network are refused. A dev server you started yourself cannot be browsed or called by the agent; have the agent start it with its own commands. To let the browser open a server the agent started in the container, set `browser.allow_private_urls: true`: there `localhost` is the container's, not yours.
- **No `domyn` commands**, no `apt`, and no root. `pip` and `npm install -g` work, and what is installed comes back when the container is recreated.
- **Environment variables** are only the ones you list under `allowed_env`, as on `local`.
- **Background commands stop** when the session ends or the conversation changes.
- **The network** is `allowed_network`: public HTTPS sites by default.

What stays the same: the agent can read and write the workspace, its skills, and `AGENTS.md`, `SOUL.md`, `.agents` and `.domyn` in your project, and every action still asks for your approval first. The built-in `domyn-cli` skill knows these differences and tells the agent about them.
:::

What the container can see:

- **The workspace**, readable and writable, at the same path as on your machine. A file a command or a tool writes is a file in your project, owned by you.
- **The skill folders**, readable and writable, so built-in skills run their scripts there and the agent can edit its own.
- **Nothing else.** Not your home folder, your credentials, your approvals, the Docker socket or your environment variables, except the names you list under `allowed_env`.
- **Two things read-only**, because a write there would reach your machine afterwards: `.git/hooks` and `.git/config` (a planted hook runs at your next `git commit`, and `git remote add` fails for the same reason), and your `config.yaml`, which the container can read. A path is masked only if it exists when the container starts. `AGENTS.md`, `SOUL.md`, `.agents` and `.domyn` in your project are writable, as they are on your machine.

What is in the container: Python with `pip`, Node with `npm`, `git` and `curl`, the same `domyn` as the one you run (the file tools run from it), and agent-browser with Chrome. There is no root, so no `apt`; the agent can install packages with `pip` and `npm`. The file tools reach the workspace, the skill folders and the container's own `/tmp`, where downloads land.

### What commands can reach on the network

One setting, `allowed_network` under `docker:` in the cli tools bundle of your `config.yaml`, says what a command can reach:

```yaml
docker:
  allowed_network: public          # the default
```

```yaml
docker:
  allowed_network:                 # a list replaces `public`: only these
  - github.com
  - "*.example.com"
```

- **`public`** (the default): any public site over HTTPS. `curl https://domyn.com`, `git clone https://github.com/...`, `pip` and `npm` all work. Addresses on your machine or your network (`localhost`, `192.168.x.x`, cloud metadata) are refused, and so are plain `http://` and any port other than 443.
- **A list of hosts**: only those, **plus the package registries** (`pypi.org`, `files.pythonhosted.org`, `registry.npmjs.org`), so `pip` and `npm` keep working. A list replaces `public`; they are not combined. An entry is a host (`github.com`) or `*.example.com` for its subdomains, HTTPS on port 443 only.
- **`none`**: no network at all.

A command that fails because the container refused a connection says which host. With a list, the fix is to add that host to it; the next command recreates the container, or waits if a background command is running.

What this protects: your machine and your network, always. With `public`, it does not stop a command from sending what it can read (files in the workspace, the variables you listed under `allowed_env`) to a public site, which is what a prompt injection would try. If that matters to you, use a list.

How it behaves:

- **One container per conversation.** It starts the first time a tool needs it, and `domyn run` prepares it while you type. The first run pulls the image, published for each release at `ghcr.io/domynai/domyn-docker-backend` (a few hundred MB, once; the session says so while it does). A build between releases (a checkout, an editable install) pulls the `latest` image, the newest release, which is what a pip or curl install gives.
- **It ends with the conversation.** `/clear`, `/resume` to another conversation and exiting stop and remove the container, and every command still running in it stops too. The session tells you when that is the case.
- **Packages come back.** What the agent installs with `pip`, and globally with `npm install -g`, is recorded in `conversations/<conversation_id>/` (`requirements.txt` and `npm-global.txt`) next to your conversation history, like the virtual environment of a conversation that runs on this machine, and installed again when the container is recreated or the conversation is resumed. A local `npm install` writes `node_modules` in your project, which stays.
- **Changed skills reach it.** Edits to files in a skill folder are seen at once. A skill folder added in the configuration, or replaced by `domyn setup`, makes the session recreate the container, unless a background command is running, in which case it waits.
- **A container that goes away is replaced.** If Docker restarts or someone removes the container while a session is running, the next command starts a new one and the session says so. Files in the workspace and the packages the agent installed come back; anything it wrote elsewhere inside the container is gone. Docker occasionally fails a start when the machine is busy: the session tries once more before reporting an error.
- **Approvals are unchanged.** They run in your terminal before the command is sent, so `/auto` and the deny rules work as before.
- **The browser is in the container too**, with its own Chrome, so pages render there and not on your machine. Its network is the same `allowed_network`.
- **Another image.** To try an image a maintainer pushed under a tag, without editing your configuration, set the environment variable `DOMYN_BACKEND_IMAGE_TAG` to that tag (`my-test`); it replaces the image of your release, and is pulled on every start so a tag pushed again is not served from an old copy. It works from a `.env` file too. **Your own image**: set `docker.image`, which wins over the variable. It needs `bash`, `timeout` and `ps`, and for the file and browser tools a `domyn` at `/opt/domyn` and agent-browser: start from `docker/backend.Dockerfile` in the repository.

### Browsing

The `browser` set drives a real browser through [agent-browser](https://github.com/vercel-labs/agent-browser). In the container backend it and Chrome are in the image and need nothing from you. On your machine, the standalone build of domyn includes it. Otherwise install it yourself, with `npm install -g agent-browser` or `brew install agent-browser`; the setup wizard warns you when it is not on your `PATH`.

The first time the agent uses a browser tool and no Chrome or Chromium is found, domyn downloads Chrome for Testing (about 200 MB) into `~/.domyn/browsers/`, once, and says so. A Chrome you already have is used as it is. To use another browser, set `browser.executable_path`. On Linux, Chrome also needs system libraries (`libnss3` and a few others, or `agent-browser install --with-deps`).

The agent reads the text of the open page or the list of elements on it, scrolls and goes back without asking. Opening a page waits for your approval, which names the site and the address, such as `https://github.com :: open 'https://github.com/docs'`. Clicking, typing and pressing a key wait for it too, naming the site and the element, such as `https://github.com :: click 'Sign in' [@e5]`. Before an approved action runs, the session checks that the browser is still on that page and that the element still has that name; if anything changed, nothing runs and the agent is told why. When the agent types into a field whose name looks like a password, a token or a card number, the approval shows how many characters it types, not the text. An address is not shortened that way: the approval shows it as the agent wrote it, parameters such as `?token=…` included, and a saved approval keeps it whole, so approving one address does not cover the same page with other parameters.

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

The pattern choice appears for commands and browser actions. For a command it proposes a pattern such as `$ git *`, which you can edit before saving; for the browser, the whole site, such as `https://github.com :: *`. A pattern covers the kind of action it was approved on: approving it on a click covers every click on that site, not typing there or opening its pages, which ask the first time they come up. A command made of several parts, such as `git pull && make`, is covered only when every part matches an approved pattern, so approving `$ git *` never approves `git status && rm -rf build`. Commands that run other commands, such as `bash`, `sudo` or `xargs`, never get a pattern proposed.

Approvals that last beyond the conversation are saved in your home directory, never in the project: `~/.domyn/approvals.yaml` for "always", and `~/.domyn/projects/` for a project. `/approvals` lists and removes them.

`/auto`, or Shift+Tab, skips every approval until you turn it off the same way. It is remembered: the next session starts with it still on, and says so. It is saved per [project](configuration.md#projects), or for everyone using `~/.domyn` itself when you use no project, so turning it on in one project does not turn it on in another. Shift+Tab works at the prompt, while the agent is working, and on an open approval question. Use it only when you trust what the agent is about to do. A short list of commands is refused even then, such as deleting your home or root directory, formatting a disk, or piping a downloaded script into a shell. That list catches accidents; it is not a security boundary.

For approvals in your own agents, see [Tool approvals](../guides/approvals.md).

## Next steps

- [Configuring the CLI](configuration.md): change the tools, the workspace, what commands may see and where the browser may go.
- [Sessions and commands](sessions.md): `/approvals`, `/auto`, `/ps` and the other session commands.
