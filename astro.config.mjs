import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import rehypeExternalLinks from 'rehype-external-links';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://XiaYiHann.github.io',
  base: '/AIInfraGuide',
  integrations: [tailwind(), sitemap()],
  check: {
    // pagefind 索引是构建产物（public/pagefind、dist/pagefind），不参与类型检查；
    // 里面有 minified 单行 JS + 数千 fragment 文件，扫进去会让 astro check 内存爆掉（2026-08-19 实证：4.27G OOM）
    exclude: ["public/**", "dist/**"],
  },
  experimental: {
    contentLayer: true,
  },
  markdown: {
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
      defaultColor: 'light',
      wrap: false,
    },
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeSlug,
      [rehypeAutolinkHeadings, { behavior: 'wrap' }],
      [rehypeExternalLinks, { target: '_blank', rel: ['nofollow', 'noopener'] }],
      rehypeKatex,
    ],
  },
});
