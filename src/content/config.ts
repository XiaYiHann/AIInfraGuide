import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const guides = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './docs/guides' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    category: z.enum([
      'learning-path',
      'prerequisites',
      'cuda-optimization',
      'distributed-training',
      'inference-optimization',
      'performance-analysis',
      'course',
    ]),
    order: z.number().default(0),
    chapter: z.number().optional(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './docs/posts' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    heroImage: z.string().optional(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
    ref: z.string().optional(),
  }),
});

const interview = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './docs/interview' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    company: z.string(),
    tier: z.enum(['T0', 'T1', 'T2', 'T3', 'T4', 'T5', '综合']),
    interviewType: z.enum(['实习', '校招', '社招', '未知']).default('未知'),
    round: z.string().optional(),
    order: z.number().default(0),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});

/** Paper 精读:用 tech-report-writing 产出的论文/博客解读 */
const papers = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './docs/papers' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    /** 原文链接 */
    originalUrl: z.string(),
    /** 来源类型:论文 / 博客 / 官方文档 */
    sourceType: z.enum(['paper', 'blog', 'docs']),
    /** 原文作者 */
    originalAuthor: z.string().optional(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
    /** 精读路线(可选):阶段、站内顺序、前置文章 id、预计时长(分钟)、难度 1-3 */
    stage: z.enum(['intuition', 'engine', 'speculative', 'attention', 'distributed', 'production']).optional(),
    order: z.number().optional(),
    prereqs: z.array(z.string()).default([]),
    minutes: z.number().optional(),
    difficulty: z.number().min(1).max(3).optional(),
  }),
});

export const collections = { guides, posts, interview, papers };
