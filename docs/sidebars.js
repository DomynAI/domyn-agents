// @ts-check

/** A non-clickable section heading, styled by .sidebar-section-title in custom.css. */
const section = (value) => ({
  type: 'html',
  value,
  className: 'sidebar-section-title',
  defaultStyle: false,
});

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docsSidebar: [
    section('Getting started'),
    'index',
    'getting-started/installation',
    'getting-started/quickstart-cli',
    'getting-started/quickstart-library',

    section('Using the CLI'),
    'cli/sessions',
    'cli/tools-and-approvals',
    'cli/configuration',

    section('Building agents'),
    'guides/tools',
    'guides/orchestration',
    'guides/remote-agents',
    'guides/llm-providers',
    'guides/transcription',
    'guides/planner-strategies',
    'guides/configuration',

    section('Shared features'),
    'guides/skills',
    'guides/standing-instructions',
    'guides/approvals',
    'guides/decision-models',
    'guides/context-compaction',
    'guides/conversations-and-memory',

    section('Running agents in an application'),
    'guides/events-and-streaming',
    'guides/hooks',
  ],

  referenceSidebar: [
    section('Reference'),
    'reference/cli',
    'reference/yaml-schema',
    'reference/events',
  ],
};

module.exports = sidebars;
