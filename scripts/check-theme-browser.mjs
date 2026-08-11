import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const baseUrl = (process.env.THEME_PREVIEW_URL || 'http://localhost:4323/AIInfraGuide').replace(/\/$/, '');
const outputDir = join(process.cwd(), 'artifacts/theme-full-site/browser');
const chromiumBinary = process.env.CHROMIUM_BIN || 'chromium';
const debuggingPort = Number(process.env.THEME_CDP_PORT || 9333);

const pages = [
  ['home', '/'],
  ['module', '/prerequisites/'],
  ['guide-4-8', '/prerequisites/模块一-前置知识/pytorch/48-benchmark与profiler/'],
  ['guide-4-12', '/prerequisites/模块一-前置知识/pytorch/412-minigpt综合项目/'],
  ['interview-index', '/interview/'],
  ['interview-article', '/interview/美团-ai-infra-一面/'],
  ['paper-index', '/papers/'],
  ['paper-article', '/papers/kimi-k3-notes/'],
  ['blog-index', '/blog/'],
  ['blog-article', '/blog/update-log/'],
  ['about', '/about/'],
  ['roadmap', '/about/roadmap/'],
  ['404', '/404.html'],
];

const modes = [
  ['light-1440', 1440, 1100, false],
  ['dark-1440', 1440, 1100, true],
  ['light-390', 390, 844, false],
  ['dark-390', 390, 844, true],
];

const selectedPages = process.env.THEME_BROWSER_PAGE
  ? pages.filter(([name]) => process.env.THEME_BROWSER_PAGE.split(',').includes(name))
  : pages;
const selectedModes = process.env.THEME_BROWSER_MODE
  ? modes.filter(([name]) => process.env.THEME_BROWSER_MODE.split(',').includes(name))
  : modes;

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitForJson(url, attempts = 80) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return response.json();
    } catch {
      // Chromium is still starting.
    }
    await delay(100);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

class CdpClient {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.nextId = 1;
    this.pending = new Map();
    this.events = new Map();
  }

  async connect() {
    await new Promise((resolve, reject) => {
      this.socket.addEventListener('open', resolve, { once: true });
      this.socket.addEventListener('error', reject, { once: true });
    });
    this.socket.addEventListener('message', ({ data }) => {
      const message = JSON.parse(data);
      if (message.id) {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        if (message.error) pending.reject(new Error(message.error.message));
        else pending.resolve(message.result);
        return;
      }
      const listeners = this.events.get(message.method) || [];
      this.events.delete(message.method);
      listeners.forEach((listener) => listener(message.params));
    });
  }

  send(method, params = {}) {
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  once(method, timeoutMs = 10000) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error(`Timed out waiting for ${method}`)), timeoutMs);
      const listener = (params) => {
        clearTimeout(timeout);
        resolve(params);
      };
      const listeners = this.events.get(method) || [];
      listeners.push(listener);
      this.events.set(method, listeners);
    });
  }

  close() {
    this.socket.close();
  }
}

function pageUrl(path) {
  return path === '/' ? `${baseUrl}/` : `${baseUrl}${path}`;
}

async function evaluate(cdp, expression) {
  const response = await cdp.send('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (response.exceptionDetails) {
    throw new Error(response.exceptionDetails.text || 'Runtime evaluation failed');
  }
  return response.result.value;
}

async function navigate(cdp, url) {
  const loaded = cdp.once('Page.loadEventFired');
  await cdp.send('Page.navigate', { url });
  await loaded;
  await delay(450);
}

async function configureMode(cdp, width, height, dark) {
  await cdp.send('Emulation.setDeviceMetricsOverride', {
    width,
    height,
    deviceScaleFactor: 1,
    mobile: width < 600,
    screenWidth: width,
    screenHeight: height,
  });
  await cdp.send('Emulation.setEmulatedMedia', {
    media: 'screen',
    features: [{ name: 'prefers-color-scheme', value: dark ? 'dark' : 'light' }],
  });
  await evaluate(
    cdp,
    `localStorage.setItem('site-theme', 'editorial'); localStorage.setItem('color-theme', '${dark ? 'dark' : 'light'}'); true`,
  );
}

async function capture(cdp, fileName) {
  const { data } = await cdp.send('Page.captureScreenshot', {
    format: 'png',
    fromSurface: true,
    captureBeyondViewport: false,
  });
  writeFileSync(join(outputDir, fileName), Buffer.from(data, 'base64'));
}

async function inspectPage(cdp) {
  return evaluate(
    cdp,
    `(() => {
      const ids = [...document.querySelectorAll('[id]')].map((node) => node.id);
      const duplicateIds = [...new Set(ids.filter((id, index) => ids.indexOf(id) !== index))];
      const overflowingElements = [...document.querySelectorAll('body *')]
        .filter((node) => !node.closest('#mobile-menu'))
        .map((node) => ({ node, rect: node.getBoundingClientRect() }))
        .filter(({ rect }) => rect.left < -1 || rect.right > window.innerWidth + 1)
        .slice(0, 12)
        .map(({ node, rect }) => ({
          element: node.tagName.toLowerCase(),
          id: node.id || null,
          className: typeof node.className === 'string' ? node.className.slice(0, 160) : null,
          text: node.textContent?.trim().replace(/\s+/g, ' ').slice(0, 100) || null,
          left: Math.round(rect.left),
          right: Math.round(rect.right),
          width: Math.round(rect.width),
          scrollWidth: node.scrollWidth,
        }));
      const wideContainers = [...document.querySelectorAll('body *')]
        .filter((node) => !node.closest('#mobile-menu'))
        .filter((node) => node.scrollWidth > node.clientWidth + 1)
        .slice(0, 12)
        .map((node) => {
          const rect = node.getBoundingClientRect();
          const style = getComputedStyle(node);
          return {
            element: node.tagName.toLowerCase(),
            id: node.id || null,
            className: typeof node.className === 'string' ? node.className.slice(0, 160) : null,
            text: node.textContent?.trim().replace(/\s+/g, ' ').slice(0, 100) || null,
            width: Math.round(rect.width),
            clientWidth: node.clientWidth,
            scrollWidth: node.scrollWidth,
            overflowX: style.overflowX,
            minWidth: style.minWidth,
          };
        });
      return {
        title: document.title,
        h1: document.querySelector('h1')?.textContent?.trim() || null,
        siteTheme: document.documentElement.dataset.siteTheme,
        colorTheme: document.documentElement.classList.contains('dark') ? 'dark' : 'light',
        noHorizontalOverflow: document.documentElement.scrollWidth <= document.documentElement.clientWidth,
        documentWidth: document.documentElement.scrollWidth,
        viewportWidth: document.documentElement.clientWidth,
        overflowingElements,
        wideContainers,
        duplicateIds,
        readingProgressCount: document.querySelectorAll('#reading-progress').length,
        backToTopCount: document.querySelectorAll('#back-to-top').length,
        errors: window.__themeBrowserErrors || [],
      };
    })()`,
  );
}

async function run() {
  mkdirSync(outputDir, { recursive: true });
  // Snap Chromium can only lock profiles in its permitted /tmp namespace.
  const profileDir = mkdtempSync('/tmp/aiinfra-theme-browser-');
  const port = debuggingPort;
  const chromium = spawn(
    chromiumBinary,
    [
      '--headless',
      '--no-sandbox',
      '--disable-gpu',
      '--hide-scrollbars',
      '--remote-allow-origins=*',
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${profileDir}`,
      'about:blank',
    ],
    { stdio: ['ignore', 'ignore', 'pipe'] },
  );
  chromium.stderr.resume();

  let cdp;
  const failures = [];
  const results = [];

  try {
    await waitForJson(`http://127.0.0.1:${port}/json/version`);
    const page = await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent('about:blank')}`, {
      method: 'PUT',
    }).then((response) => response.json());
    cdp = new CdpClient(page.webSocketDebuggerUrl);
    await cdp.connect();
    await cdp.send('Page.enable');
    await cdp.send('Runtime.enable');
    await cdp.send('Page.addScriptToEvaluateOnNewDocument', {
      source: `(() => {
        window.__themeBrowserErrors = [];
        window.addEventListener('error', (event) => window.__themeBrowserErrors.push(String(event.error || event.message)));
        window.addEventListener('unhandledrejection', (event) => window.__themeBrowserErrors.push(String(event.reason)));
        const originalError = console.error.bind(console);
        console.error = (...args) => {
          window.__themeBrowserErrors.push(args.map(String).join(' '));
          originalError(...args);
        };
      })();`,
    });

    await navigate(cdp, pageUrl('/'));

    for (const [modeName, width, height, dark] of selectedModes) {
      for (const [pageName, path] of selectedPages) {
        await configureMode(cdp, width, height, dark);
        await navigate(cdp, pageUrl(path));
        const inspection = await inspectPage(cdp);
        const record = { page: pageName, path, mode: modeName, ...inspection };
        results.push(record);
        await capture(cdp, `${pageName}-${modeName}.png`);

        const valid =
          inspection.siteTheme === 'editorial' &&
          inspection.colorTheme === (dark ? 'dark' : 'light') &&
          inspection.noHorizontalOverflow &&
          inspection.duplicateIds.length === 0 &&
          inspection.readingProgressCount === 1 &&
          inspection.backToTopCount === 1 &&
          inspection.errors.length === 0;
        if (!valid) failures.push(record);
      }
    }

    await configureMode(cdp, 1440, 1100, false);
    await navigate(cdp, `${pageUrl('/')}?theme=classic`);
    const classicQuery = await evaluate(cdp, `({ theme: document.documentElement.dataset.siteTheme, saved: localStorage.getItem('site-theme') })`);
    await evaluate(cdp, `document.getElementById('site-theme-preview').click()`);
    const headerSwitch = await evaluate(cdp, `({ theme: document.documentElement.dataset.siteTheme, saved: localStorage.getItem('site-theme') })`);
    await navigate(cdp, pageUrl('/'));
    const switchPersistence = await evaluate(cdp, `({ theme: document.documentElement.dataset.siteTheme, saved: localStorage.getItem('site-theme') })`);
    await navigate(cdp, `${pageUrl('/')}?theme=editorial`);
    const editorialQuery = await evaluate(cdp, `({ theme: document.documentElement.dataset.siteTheme, saved: localStorage.getItem('site-theme') })`);

    await evaluate(cdp, `document.querySelector('button[aria-label="搜索"]').click()`);
    await delay(1000);
    const searchOpen = await evaluate(cdp, `({
      open: document.getElementById('search-modal').classList.contains('flex'),
      inputReady: Boolean(document.querySelector('.pagefind-ui__search-input')),
      bodyLocked: document.body.style.overflow === 'hidden',
    })`);
    await capture(cdp, 'interaction-search-light-1440.png');

    await configureMode(cdp, 390, 844, false);
    await navigate(cdp, pageUrl('/'));
    await evaluate(cdp, `document.getElementById('mobile-menu-btn').click()`);
    await delay(200);
    const mobileMenu = await evaluate(cdp, `({
      open: !document.getElementById('mobile-menu').classList.contains('translate-x-full'),
      overlay: document.getElementById('mobile-menu-overlay').classList.contains('opacity-100'),
      bodyLocked: document.body.style.overflow === 'hidden',
    })`);
    await capture(cdp, 'interaction-mobile-menu-light-390.png');

    await navigate(cdp, pageUrl('/prerequisites/模块一-前置知识/pytorch/412-minigpt综合项目/'));
    await evaluate(cdp, `document.querySelector('.mobile-toc summary').click()`);
    await delay(200);
    const mobileToc = await evaluate(cdp, `({ open: document.querySelector('.mobile-toc')?.open === true })`);
    await capture(cdp, 'interaction-mobile-toc-light-390.png');

    const interactions = { classicQuery, headerSwitch, switchPersistence, editorialQuery, searchOpen, mobileMenu, mobileToc };
    if (
      classicQuery.theme !== 'classic' || classicQuery.saved !== 'classic' ||
      headerSwitch.theme !== 'editorial' || headerSwitch.saved !== 'editorial' ||
      switchPersistence.theme !== 'editorial' || switchPersistence.saved !== 'editorial' ||
      editorialQuery.theme !== 'editorial' || editorialQuery.saved !== 'editorial' ||
      !Object.values(searchOpen).every(Boolean) ||
      !Object.values(mobileMenu).every(Boolean) ||
      !mobileToc.open
    ) {
      failures.push({ interactions });
    }

    const report = {
      generatedAt: new Date().toISOString(),
      baseUrl,
      chromium: chromiumBinary,
      pageCount: selectedPages.length,
      modeCount: selectedModes.length,
      screenshotCount: selectedPages.length * selectedModes.length + 3,
      interactions,
      failures,
      results,
    };
    writeFileSync(join(outputDir, 'browser-check.json'), `${JSON.stringify(report, null, 2)}\n`);

    if (failures.length > 0) {
      console.error(`Theme browser check failed with ${failures.length} failure(s).`);
      process.exitCode = 1;
    } else {
      console.log(`Theme browser check passed: ${selectedPages.length} page types × ${selectedModes.length} modes, ${report.screenshotCount} screenshots, 7 interactions.`);
    }
  } finally {
    cdp?.close();
    chromium.kill('SIGTERM');
    rmSync(profileDir, { recursive: true, force: true });
  }
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
