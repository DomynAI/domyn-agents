import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {useLocation} from '@docusaurus/router';

type Versions = {latest: string; versions: string[]};

// Every release is a separate build under <root>/<tag>/, so switching version
// is a plain page load to the same path in the other build, never client-side
// routing.
export default function VersionDropdown(): React.JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  const {pathname} = useLocation();
  const current = String(siteConfig.customFields?.docsVersion ?? 'dev');
  const versionsUrl = (siteConfig.customFields?.versionsUrl as string | null) ?? null;
  const [listed, setListed] = React.useState<Versions | null>(null);

  React.useEffect(() => {
    if (!versionsUrl) {
      return;
    }
    let cancelled = false;
    fetch(versionsUrl)
      .then((response) => (response.ok ? response.json() : null))
      .then((data: Versions | null) => {
        if (!cancelled && data && Array.isArray(data.versions)) {
          setListed(data);
        }
      })
      .catch(() => {
        // Without the list the dropdown shows only this version.
      });
    return () => {
      cancelled = true;
    };
  }, [versionsUrl]);

  const versions = listed?.versions.includes(current)
    ? listed.versions
    : [current, ...(listed?.versions ?? [])];

  const onChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const target = event.target.value;
    if (target === current || !versionsUrl) {
      return;
    }
    const root = versionsUrl.slice(0, versionsUrl.lastIndexOf('/') + 1);
    const rest = pathname.startsWith(siteConfig.baseUrl)
      ? pathname.slice(siteConfig.baseUrl.length)
      : '';
    window.location.assign(`${root}${target}/${rest}`);
  };

  return (
    <select
      className="navbar__item version-dropdown"
      aria-label="Documentation version"
      value={current}
      onChange={onChange}
      disabled={versions.length < 2}>
      {versions.map((version) => (
        <option key={version} value={version}>
          {version === listed?.latest ? `${version} (latest)` : version}
        </option>
      ))}
    </select>
  );
}
