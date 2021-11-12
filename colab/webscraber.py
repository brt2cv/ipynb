#!/usr/bin/env python3
# @Date    : 2021-11-12
# @Author  : Bright (brt2@qq.com)
# @Link    : https://gitee.com/brt2

# %%
import json
import requests
from bs4 import BeautifulSoup

# %%
def get_html(url):
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        r.encoding = r.apparent_encoding  # 防止乱码
        return r.text
    except:
        return ""

# %%
def parse_selector(html, selector):
    soup = BeautifulSoup(html, "html.parser")
    data = soup.select(selector["selector"])
    print(data)
    return data


s = sitemap["selectors"][0]
print("....", s)
parse_selector(html, s)



# %%
import ipyenv as uu
uu.chdir(__file__)

with open(uu.rpath("sitemap.json")) as fp:
    sitemap = json.load(fp)
    print(sitemap)

# %%
html = get_html(sitemap["startUrl"][0])  # startUrl列表仅仅一个元素
print(html)
# %%
