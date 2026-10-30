# Security policy

## Reporting a vulnerability

Please do not report security vulnerabilities in public issues, discussions or pull requests.

Report them privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability** ([direct link](https://github.com/domynai/domyn-agents/security/advisories/new)). If you cannot use GitHub, write to [security@domyn.com](mailto:security@domyn.com). Include what you found, how to reproduce it, and the version (`domyn --version`) or commit you tested.

We will confirm that we received your report, keep you informed while we work on a fix, and credit you in the advisory unless you prefer otherwise. Please give us a reasonable time to release the fix before disclosing it.

## Supported versions

Fixes go into the next release, from the latest stable one. Older releases, release candidates and the nightly alphas are not patched.

## Scope

Domyn Agents runs model output as tool calls on your machine, so some areas matter more than others:

- the built-in file and shell tools, and the `Workspace` that confines them;
- the approval gate: a call running without the approval its tool requires, or a grant covering more than it says;
- credentials: API keys stored by `domyn setup`, OAuth tokens for MCP servers, secrets in YAML configurations;
- anything that lets content read by the agent (a file, a web page, a tool result) change what it is allowed to do.

A model that is talked into asking for something harmful is not a vulnerability in itself. A harmful call that runs without the approval it should have needed is.
