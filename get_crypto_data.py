from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import pandas as pd
import json

key = "1f37dc1a-4cf1-4079-916b-1e03bcf7912c"

url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
    'start':'1',
    'limit':'5000',
    'convert':'USD'
}
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': key,
}

session = Session()
session.headers.update(headers)

try:
    response = session.get(url, params=parameters)
    data = json.loads(response.text)
    data = data['data']

    cryptocurrency = []
    for crypto in data:
        market_cap = crypto["quote"]["USD"]["market_cap"]
        if market_cap != None:
            cryptocurrency.append([crypto['symbol'].lower(), crypto["quote"]["USD"]['market_cap']])
        else:
            continue

    cryptocurrency.sort(key=lambda x: x[1], reverse=True)

    cryptocurrency = cryptocurrency[:50]
    df = pd.DataFrame(data=cryptocurrency, columns=["symbol", "market_cap"])

    df.to_csv("selected_crypto.csv", index=False)

except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)
