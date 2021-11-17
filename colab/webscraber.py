#!/usr/bin/env python3
# @Date    : 2021-11-12
# @Author  : Bright (brt2@qq.com)
# @Link    : https://gitee.com/brt2

# %%
import ipyenv as uu
uu.chdir(__file__)
with open(uu.rpath("sitemap.json")) as fp:
# with open("sitemap.json") as fp:
    import json
    sitemap = json.load(fp)

# %%
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException

def make_driver(headless=False, wait_time=10):
    if headless:
        from selenium.webdriver.chrome.options import Options

        options = webdriver.ChromeOptions()
        options.set_headless()
        driver = webdriver.Chrome(chrome_options=options)
    else:
        driver = webdriver.Edge(executable_path=r"D:\Home\bin\webdriver\msedgedriver.exe")

    driver.implicitly_wait(wait_time)
    return driver

engine = make_driver()

# %%
import queue

class Schedule:
    def __init__(self, dict_sitemap):
        self._sitemap = dict_sitemap
        # {'_id': 'huawei-phone',
        # 'startUrl': ['https://consumer.huawei.com/cn/phones/?ic_medium=hwdc&ic_source=corp_header_consumer'],
        # 'selectors': [
        #   {'id': 'subpage', 'parentSelectors': ['_root'], 'type': 'SelectorLink', 'selector': 'a.product-block__title-link', 'multiple': True, 'delay': 0},
        #   {'id': 'visit-params', 'parentSelectors': ['subpage'], 'type': 'SelectorLink', 'selector': "a[title='规格参数']", 'multiple': False, 'delay': 0},
        #   {'id': 'weight', 'parentSelectors': ['visit-params'], 'type': 'SelectorText', 'selector': 'div:nth-of-type(2) div.large-accordion__wrap:nth-of-type(1) p', 'multiple': False, 'delay': 0, 'regex': ''},
        #   {'id': 'size', 'parentSelectors': ['visit-params'], 'type': 'SelectorText', 'selector': 'div.large-accordion__inner--short:nth-of-type(1) > p:nth-of-type(1)', 'multiple': False, 'delay': 0, 'regex': ''}
        # ]}
        self._selector_graph = self._make_graph(self._sitemap["selectors"])
        # {'_root': {'subpage': {'visit-params': {'weight': {}, 'size': {}}}}}
        self._node_map = self._make_map(self._sitemap["selectors"])
        # {'_root': {'subpage'},
        #  'subpage': {'visit-params'},
        #  'visit-params': {'size', 'weight'}
        # }
        self._dict_priority = {}
        self._make_priority(self._selector_graph)
        # {'_root': 0, 'subpage': -1, 'visit-params': -2, 'weight': -3, 'size': -3}
        self._dict_node = {d["id"]: d for d in self._sitemap["selectors"]}
        # {'subpage': {'id': 'subpage', 'parentSelectors': ['_root'], 'type': 'SelectorLink', 'selector': 'a.product-block__title-link', 'multiple': True, 'delay': 0},
        #  'visit-params': {'id': 'visit-params', 'parentSelectors': ['subpage'], 'type': 'SelectorLink', 'selector': "a[title='规格参数']", 'multiple': False, 'delay': 0},
        #  'weight': {'id': 'weight', 'parentSelectors': ['visit-params'], 'type': 'SelectorText', 'selector': 'div:nth-of-type(2) div.large-accordion__wrap:nth-of-type(1) p', 'multiple': False, 'delay': 0, 'regex': ''},
        #  'size': {'id': 'size', 'parentSelectors': ['visit-params'], 'type': 'SelectorText', 'selector': 'div.large-accordion__inner--short:nth-of-type(1) > p:nth-of-type(1)', 'multiple': False, 'delay': 0, 'regex': ''}
        # }
        self._make_task()

    def _make_graph(self, list_tree: List[dict]) -> dict:
        dict_tree = {i["id"]: {} for i in list_tree}
        dict_tree["_root"] = {}

        for node in list_tree:
            node_id = node.get("id")
            parent = node.get("parentSelectors")[0]
            dict_tree[parent][node_id] = dict_tree[node_id]

        return {"_root": dict_tree["_root"]}

    def _make_priority(self, graph, priority=0):
        for key, d in graph.items():
            self._dict_priority[key] = priority
            self._make_priority(d, priority-1)

    def _make_map(self, list_tree: List[dict]) -> dict:
        dict_tree = {}
        for node in list_tree:
            node_id = node.get("id")
            parent = node.get("parentSelectors")[0]
            if parent not in dict_tree:
                dict_tree[parent] = set()
            dict_tree[parent].add(node_id)
        return dict_tree

    def get_children(self, parent_node):
        """ return set of children in self._node_map """
        return self._node_map[parent_node]

    def _make_task(self):
        import queue
        self.task = queue.PriorityQueue()  # {"_root": self._sitemap["startUrl"][0]}
        self.put_task(self.get_children("_root"), self._sitemap["startUrl"][0])  # 初始化任务列表

    def get_node(self, node_id: str):
        return self._dict_node[node_id]

    def put_task(self, set_node, url):
        for node in set_node:
            task = [url, node]
            self.task.put((self._dict_priority[node], task))

    def get_task(self):
        """ 获取下一个任务, return: (url, ) """
        priority, task = self.task.get(block=False, timeout=1)
        return task

class Spider:
    def __init__(self, driver, schedule):
        self.driver = driver
        self.schedule = schedule
        self.curr_url = None

    def open_url(self, url):
        self.driver.get(url)
        self.curr_url = url

    def _click_link(self, elem):
        # try:
        #     elem.click()
        # except StaleElementReferenceException:
        # ActionChains
        from selenium.webdriver.common.action_chains import ActionChains
        ActionChains(self.driver).move_to_element(elem).click(elem).perform()

    # def _get_link(self):
    #     pass
    # def _get_text(self):
    #     pass

    def parse_node(self, node):
        # 获取选择器
        css_path = node["selector"]
        if node["multiple"]:
            list_elems = self.driver.find_elements("css selector", css_path)
        else:
            elem = self.driver.find_element("css selector", css_path)
            list_elems = [elem]

        # 根据类型执行操作
        if node["type"] == "SelectorLink":
            for elem in list_elems:
                link_url = elem.get_attribute("href")
                self.schedule.put_task(self.schedule.get_children(node["id"]), link_url)
        elif node["type"] == "SelectorText":
            print("--- ", elem.text)
        else:
            raise NotImplementedError()

    def run(self):
        while True:
            try:
                url, node_id = self.schedule.get_task()
                node = self.schedule.get_node(node_id)
                # print(">>> ", url)
                if url != self.curr_url:
                    self.driver.get(url)
                self.parse_node(node)
            except NoSuchElementException as e:
                print("【err】FindElement Failed:", node)
                continue
            except queue.Empty as e:
                # print("【err】", e)
                break
        print("Well done")

s = Schedule(sitemap)
e = Spider(engine, s)
e.run()
engine.quit()
