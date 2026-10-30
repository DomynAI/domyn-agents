// @ts-check
// Published on GitHub Pages by .github/workflows/docs.yml, which sets SITE_URL,
// BASE_URL and DOCS_VERSION: every release tag is its own build under
// <repo>/<tag>/, and <repo>/versions.json lists them all for the navbar's
// version dropdown. The defaults are for `make serve`.

const {themes} = require('prism-react-renderer');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Domyn Agents',
  tagline: 'Documentation',
  favicon: 'img/favicon.ico',

  url: process.env.SITE_URL ?? 'http://localhost:3000',
  baseUrl: process.env.BASE_URL ?? '/',

  organizationName: 'Domyn',
  projectName: 'domyn-agents-docs',

  onBrokenLinks: 'throw',

  customFields: {
    // The tag this build is, or 'dev' (main, pull requests, local).
    docsVersion: process.env.DOCS_VERSION ?? 'dev',
    // Where versions.json lives: one level above a tag's baseUrl.
    versionsUrl: process.env.VERSIONS_URL ?? null,
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          breadcrumbs: false,
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],

  themeConfig: {
    colorMode: {
      disableSwitch: true,
      respectPrefersColorScheme: true,
      defaultMode: 'light',
    },

    image: 'img/social-card.png',

    navbar: {
      logo: {
        alt: 'Domyn Logo',
        src: 'img/logo.svg',
        srcDark: 'img/logo-white.svg',
        height: 24,
      },
      items: [
        {
          type: 'custom-versionDropdown',
          position: 'right',
        },
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          label: 'Docs',
          position: 'right',
        },
        {
          type: 'docSidebar',
          sidebarId: 'referenceSidebar',
          label: 'Reference',
          position: 'right',
        },
        {
          href: 'https://github.com/domynai/domyn-agents',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    prism: {
      theme: themes.github,
      darkTheme: themes.dracula,
      additionalLanguages: ['bash', 'yaml'],
    },
  },

  themes: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['en'],
        indexBlog: false,
        docsRouteBasePath: '/',
      },
    ],
  ],
};

module.exports = config;
