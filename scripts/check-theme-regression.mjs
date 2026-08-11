import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const distDir = join(projectRoot, 'dist');
const defaultBaseline = join(projectRoot, 'artifacts/theme-full-site/url-baseline.json');

function collectHtmlFiles(directory) {
  const files = [];
  const visit = (current) => {
    for (const name of readdirSync(current)) {
      const absolute = join(current, name);
      if (statSync(absolute).isDirectory()) visit(absolute);
      else if (name.endsWith('.html')) files.push(relative(directory, absolute).replaceAll('\\', '/'));
    }
  };
  visit(directory);
  return files.sort((a, b) => a.localeCompare(b, 'zh-CN'));
}

if (!existsSync(distDir)) {
  console.error('dist/ is missing. Run npm run build before the theme regression check.');
  process.exit(1);
}

const pages = collectHtmlFiles(distDir);
const writeIndex = process.argv.indexOf('--write-baseline');

if (writeIndex !== -1) {
  const requestedPath = process.argv[writeIndex + 1];
  const baselinePath = requestedPath ? resolve(projectRoot, requestedPath) : defaultBaseline;
  const commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: projectRoot, encoding: 'utf8' }).trim();
  mkdirSync(dirname(baselinePath), { recursive: true });
  writeFileSync(
    baselinePath,
    `${JSON.stringify({ generatedFromCommit: commit, pageCount: pages.length, pages }, null, 2)}\n`,
    'utf8',
  );
  console.log(`Theme URL baseline written: ${pages.length} pages from ${commit.slice(0, 7)}.`);
  process.exit(0);
}

const baselinePath = process.argv[2] ? resolve(projectRoot, process.argv[2]) : defaultBaseline;
if (!existsSync(baselinePath)) {
  console.error(`Theme URL baseline is missing: ${relative(projectRoot, baselinePath)}`);
  process.exit(1);
}

const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
const expected = new Set(baseline.pages);
const actual = new Set(pages);
const missing = baseline.pages.filter((page) => !actual.has(page));
const extra = pages.filter((page) => !expected.has(page));

if (baseline.pageCount !== pages.length || missing.length > 0 || extra.length > 0) {
  console.error(`Theme URL regression failed: expected ${baseline.pageCount}, found ${pages.length}.`);
  if (missing.length) console.error(`Missing (${missing.length}): ${missing.slice(0, 20).join(', ')}`);
  if (extra.length) console.error(`Extra (${extra.length}): ${extra.slice(0, 20).join(', ')}`);
  process.exit(1);
}

console.log(`Theme URL regression passed: ${pages.length} HTML pages match ${baseline.generatedFromCommit.slice(0, 7)}.`);
