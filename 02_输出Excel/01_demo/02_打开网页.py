from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import json


def getVersionMap():
	# 等待网页加载完毕
	wait = WebDriverWait(browser, 10)
	wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'k-content')))
	# 获取所有class为k-content的div元素
	# 获取目标内容
	return browser.execute_script(
		"return $('.k-content').eq(2).find('p').map((idx, val) => { return {val:/\S+/g.exec(val.innerText)[0], idx}})")


def printJson():
	global name
	browser.get(url)
	# [{idx:0, val:V5.0}]
	versionMap = getVersionMap()
	arr = {}
	for versionItem in versionMap:
		version = versionItem['val']
		isV = version.startswith('V')
		if not isV:
			continue
		idx = versionItem['idx']
		browser.execute_script(
			f"$('.k-content').eq(2).find('p').eq({idx}).click();")
		# 等待新的内容加载完成
		wait = WebDriverWait(browser, 10)
		wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.publish")))
		valList = browser.execute_script(
			'''return $('li.publish').map((idx, val) => {
    return {
        name: $(val).find('.publish-span').text(),
        href: $(val).find('.publish-a').attr('href')
    }
})''')
		arr[version] = valList
		print(version)
	json_str2 = json.dumps(arr, ensure_ascii=False)
	# 将 JSON 格式字符串写入文件
	with open(name, "w", encoding="utf-8") as f:
		f.write(json_str2)
	browser.exit()
	print(versionMap)


if __name__ == '__main__':
	# 打开浏览器并加载网页
	chromedriver = '/Users/judy/Downloads/chr/chromedriver'

	# 启动Chrome浏览器
	browser = webdriver.Chrome(executable_path=chromedriver)

	# 打开网页
	# url = "https://developer.kingdee.com/knowledge/specialDetail/174931520225092096?category=181160777380834048&id=307655665731349760&productLineId=29"
	# name = "./集成服务云.json"

	# printJson()

	# 打开网页
	# url = "https://developer.kingdee.com/knowledge/specialDetail/174931520225092096?category=181160777515051776&id=447460388506859520&productLineId=29"
	# name = "./开发平台.json"
	#
	# printJson()

	# 打开网页
	url = "https://developer.kingdee.com/knowledge/specialDetail/174931520225092096?category=187214294709708288&id=447460388792072192&productLineId=29"
	name = "./流程服务云.json"

	printJson()


