# %%
import os.path
import json

PATH_UPLOAD = r"D:\Home\workspace\daily\2021-11.km"
PATH_COOKIE = r"D:\Home\workspace\daily\cookies.json"
BASENAME = os.path.basename(PATH_UPLOAD)

# %%
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains

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

def select_css(selector, click=False, double_click=False):
    try:
        elem = engine.find_element("css selector", selector)
    except NoSuchElementException as e:
        print("【NoSuchElementException】", selector)
        raise e

    if click:
        try:
            elem.click()
        except StaleElementReferenceException:
            ActionChains(engine).move_to_element(elem).click(elem).perform()
    elif double_click:
        ActionChains(engine).move_to_element(elem).double_click(elem).perform()

    return elem

engine = make_driver()
engine.get("https://cloud.huawei.com/")

# %% 加载cookies
# with open(PATH_COOKIE, "r") as fp:
#     cookies = json.load(fp)
#     engine.add_cookie(cookies[0])
#     engine.refresh()

# %% 登录
def login():
    engine.switch_to.frame("frameAddress")  # 请注意切换到iframe
    elems = engine.find_elements("css selector", "input.hwid-input")  # 获取到两个元素
    account_info = ("18131218231", "brt325817")
    for elem, val in zip(elems, account_info):
        elem.send_keys(val)

    select_css("div.hwid-btn.hwid-btn-primary.hwid-login-btn", click=True)

login()
# %%
# 导入cookies：验证无法解决问题
# t = {'domain': '.cloud.huawei.com', 'httpOnly': False, 'name': 'loginSecLevel', 'path': '/', 'secure': True, 'value': '2'}
# engine.add_cookie(t)
# engine.refresh()

# %%
# 退出后再登录，无需验证
# select_css("div.userBox", click=True)
# select_css("li.exit", click=True)
engine.refresh()  # 直接刷新即可实现二次登录
login()

# %%
# 存储cookie
# cookies = engine.get_cookies()
# with open(PATH_COOKIE, "w") as fp:
#     json.dump(cookies, fp)

# %% 进入云盘
import time
time.sleep(3)

# select_css(".warpHome.ncollection", click=True)
engine.get("https://cloud.huawei.com/home#/collection/v2/all")

time.sleep(1)

# 关闭“体验”弹窗
try:
    # experience = select_css("div.sExperience")
    # ActionChains(engine).move_to_element(experience).click(experience).perform()
    select_css("span.sExperienceText", click=True)
except NoSuchElementException:
    pass

# %% 进入daily目录
select_css('span.fileNameText[title="daily"]', double_click=True)

# %% 删除旧文件
list_elems = engine.find_elements("css selector", "div.fileLine-inline")
for elem in list_elems:
    elem.click()
    try:
        if elem.find_element("css selector", f'span.fileNameText[title="{BASENAME}"]'):
            # a = elem.find_element("css selector", 'span.fileBtn_delete[title="删除"]')
            # ActionChains(engine).move_to_element(a).click(a).perform()
            select_css("li.blueDelBtn", click=True)
            select_css("div.popButton.text-red", click=True)
            break
    except NoSuchElementException as e:
        print("NoSuchElementException: ", e)

# %% 上传文件
# select_css("li.horizontalLi.clc-btn.blueUpBtn")
# select_css("div.upLoadBox_file.upLoad_sbtn")
engine.find_element("css selector", "input#clc-uploadFile_box").send_keys(PATH_UPLOAD)

# %%
time.sleep(3)
engine.quit()
