# coding:utf-8
import requests
import json

query = '王祖贤'
'''下载图片'''


def download(src, id):
    local_dir = './' + str(id) + '.jpg'
    try:
        pic = requests.get(src, timeout=10)
        fp = open(local_dir, 'wb')
        fp.write(pic.content)
        fp.close()
    except requests.exceptions.ConnectionError:
        print("图片无法下载")


''' for 循环 请求全部的 url'''
for i in range(0, 22471, 20):
    url = 'https://www.douban.com/j/search_photo?q=' + query + '&limit=20&start=' + str(i)
    # 得到返回结果
    html = requests.get(url).text
    # 将json 转换成 python 对象
    response = json.loads(html, encoding='utf-8')
    for image in response['images']:
        # 查看当前下载的图片网址
        print(image['src'])
        # 下载一张图片
        download(image['src'], image['id'])
