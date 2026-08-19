#!/bin/sh
# 验证 AIInfraGuide 线上页面。用法: check_live.sh <site-path> [signature]
#   site-path:  以 /AIInfraGuide 开头的站点内路径（可含中文，脚本自动 percent-encode）
#   signature:  可选；正文里应出现的字符串（验证新内容真的上线了）
# 退出码: 0=200 且签名匹配; 1=HTTP 非 200; 2=200 但签名缺失
set -e
path="${1:?usage: check_live.sh <site-path> [signature]}"
sig="${2:-}"
url=$(python3 -c "
import urllib.parse, sys
print('https://xiayihann.github.io' + urllib.parse.quote(sys.argv[1], safe='/'))
" "$path")
code=$(curl -s -o /tmp/check_live_body.html -w '%{http_code}' -L "$url")
echo "HTTP $code  $url"
if [ "$code" != "200" ]; then exit 1; fi
if [ -n "$sig" ]; then
  if grep -q "$sig" /tmp/check_live_body.html; then
    echo "OK: signature '$sig' found"
    exit 0
  else
    echo "MISSING: signature '$sig' NOT found (CDN 缓存？等 1~2 分钟重试)"
    exit 2
  fi
fi
echo "OK: page is live"
exit 0
