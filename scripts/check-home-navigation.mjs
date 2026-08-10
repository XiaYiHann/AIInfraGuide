import { readFile } from 'node:fs/promises';

const base = '/AIInfraGuide';
const chapterHref = `${base}/prerequisites/模块一-前置知识/第4章-pytorch框架/`;
const capstoneHref = `${base}/prerequisites/模块一-前置知识/pytorch/412-minigpt综合项目/`;

const checks = [
  {
    file: 'dist/index.html',
    expected: [chapterHref],
    label: '首页必须提供 PyTorch 第4章直达入口',
  },
  {
    file: 'dist/prerequisites/index.html',
    expected: [chapterHref.replace(/\/$/, ''), capstoneHref.replace(/\/$/, '')],
    label: '前置知识目录必须索引第4章及综合项目',
  },
  {
    file: 'dist/guides/ai-infra学习路线/index.html',
    expected: [chapterHref.replace(/\/$/, ''), capstoneHref],
    label: '学习路线必须索引第4章及综合项目',
  },
];

const failures = [];
for (const check of checks) {
  let html;
  try {
    html = await readFile(check.file, 'utf8');
  } catch (error) {
    failures.push(`${check.label}: 无法读取 ${check.file}: ${error.message}`);
    continue;
  }

  const hrefs = [...html.matchAll(/href="([^"]+)"/g)].map((match) => {
    try {
      return decodeURI(match[1]).replace(/\/$/, '');
    } catch {
      return match[1].replace(/\/$/, '');
    }
  });
  const missing = check.expected
    .map((href) => href.replace(/\/$/, ''))
    .filter((href) => !hrefs.includes(href));
  if (missing.length > 0) {
    failures.push(`${check.label}: ${check.file} 缺少 ${missing.join(', ')}`);
  }
}

if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}

console.log('Home navigation check passed: 首页 → PyTorch 第4章 → 4.12 路径完整。');
