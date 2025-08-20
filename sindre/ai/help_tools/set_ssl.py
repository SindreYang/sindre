import os



def set_ssl_with_steam():
    """
    将steam++的证书写入到ssl中,防止requests.exceptions.SSLError报错
    """

    import requests
    import certifi
    SteamToolsCertificate_path = os.path.join(os.path.dirname(__file__),"data","SteamTools.Certificate.pfx")
    print("将steam++的证书写入到ssl中,防止requests.exceptions.SSLError报错 ,原则上只调用一次")
    try:
        print('Checking connection to Huggingface...')
        test = requests.get('https://huggingface.co')
        print('Connection to Huggingface OK.')
    except requests.exceptions.SSLError as err:
        print('SSL Error. Adding custom certs to Certifi store...')
        cafile = certifi.where()
        with open(SteamToolsCertificate_path, 'rb') as infile:
            customca = infile.read()
        with open(cafile, 'ab') as outfile:
            outfile.write(customca)
        print('That might have worked.')


def disable_ssl():
    import requests
    import warnings
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    # 禁用 SSL 验证
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)  # 忽略警告
    session = requests.Session()
    session.verify = False  # 禁用验证
    requests.Session = lambda: session  # 全局覆盖 Session
