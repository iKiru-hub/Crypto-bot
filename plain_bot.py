import os
import requests
import time

import numpy as np
import pandas as pd
import json
import scipy.stats

import ccxt
from binance.client import Client
import binance





class Trader:

    """
    the agent that carries out the trading. In a nutshell it checks whether it is a right moment to buy or sell and
    register the eventual orders
    """

    def __init__(self, capital, fees_percent, stoploss_percent=0.2):

        # trading parameters
        self.capital = capital
        self.deposited = capital
        self.fees_percent = fees_percent
        self.stoploss_percent = stoploss_percent

        # trading variables
        self.status = 'free'
        self.coin = ''
        self.buy_price = 0.
        self.sell_price = 0.
        self.stoploss_price = 0.
        self.watermark = 0.
        self.expected_profit = 0.
        self.nb_coins = 0.
        self.fees = 0.
        self.gain = 0.
        self.profit = 0.

        # record
        self.count = 0
        self.transaction = {'trade': [],
                            'time': [],
                            'coin': [],
                            'buy': [],
                            'sell': [],
                            'stop': [],
                            'fees': [],
                            'gain': [],
                            'profit': [],
                            'capital': []}



    def check_buy(self, hlcv):

        """ the place where, given hlc data (high-low-close) a pre-defined algorithm
         decides whether to buy
         """

        # to fill in

        # now the decision is random, delete this when using real money!
        return bool(np.random.random() > .7)


    def check_sell(self, hlcv):

        """ the place where, given hlc data (high-low-close) a pre-defined algorithm
         decides whether to sell
         """

        # to fill in

        # now the decision is random, delete this when using real money!
        return bool(np.random.random() > .5)


    def update_stoploss(self, price: float):

        """
        :param price: the current coin price

        update the value of the stop-loss
        """

        # expected = current_worth_value - bought_worth_value
        #          = current_price * N_coins * (1 - %) - Buy_price * N_coins * (1 + %)   | % : fees percent
        self.expected_profit = price * self.nb_coins * (1 - self.fees_percent) - self.capital * (1 + self.fees_percent)

        # if the expected profit is positive, update the stop-loss. Namely, improve the minimum profit
        if self.expected_profit > 0:

            # - Capital * beta + relu(P)
            lower_profit = - self.capital * self.stoploss_percent + self.expected_profit

            # W - Pl / (N * (1 - %))
            self.stoploss_price = self.watermark + lower_profit / (self.nb_coins * (1 - self.fees_percent))


    def buy(self, price: float, coin: str):

        """
        :param price: the current coin price
        :param coin: the coin name
        :return: the number of coins bought
        """

        # register variables
        self.buy_price = price
        self.nb_coins = self.capital / price

        # price value at which the profit is exactly zero (considering the fees)
        self.watermark = price * (1 + self.fees_percent) / (1 - self.fees_percent)
        self.coin = coin

        # current expected profit
        self.expected_profit = self.capital * (1 - self.fees_percent) / (1 + self.fees_percent)

        # minimum profit calculated considering the stoploss placed at the price
        # according to the fixed parameter percent
        lower_profit = - self.capital * self.stoploss_percent  # starts minus

        # current stoploss position
        self.stoploss_price = self.watermark + lower_profit / (self.nb_coins * (1 - self.fees_percent))

        self.status = 'charged'

        return round(self.nb_coins, 7)


    def sell(self, price: float):

        """
        :param price: current coin price
        :return: number of coins to seel
        """

        # register variables
        self.sell_price = price

        self.fees = (self.capital + self.sell_price * self.nb_coins) * self.fees_percent

        self.gain = (self.sell_price - self.buy_price) * self.nb_coins - self.fees
        self.profit += self.gain
        self.capital += self.gain

        return round(self.nb_coins, 7)


    def check_control(self, hlcv, kind):

        """ handle the buy and sell check """

        if kind == 'buy':
            return self.check_buy(hlcv=hlcv)

        elif kind == 'sell':

            price = hlcv[2][-1]
            self.update_stoploss(price=price)

            if price < self.stoploss_price:
                return True

            return self.check_sell(hlcv=hlcv)

        else:
            raise TypeError('Sorry, only <buy> or <sell> options are allowed')



    def record(self, timestamp=None):

        """
        :param timestamp: time at which the transaction occurred
        """

        """ record the transaction """

        clock = time.localtime()
        self.count += 1

        self.transaction['trade'].append(self.count)
        if timestamp is None:
            self.transaction['time'].append(f'{clock.tm_hour}:{clock.tm_min}:{clock.tm_sec}')
        else:
            self.transaction['time'].append(round(timestamp /60, 1))
        self.transaction['coin'].append(self.coin)
        self.transaction['buy'].append(self.buy_price)
        self.transaction['sell'].append(self.sell_price)
        self.transaction['stop'].append(self.stoploss_price)
        self.transaction['fees'].append(round(self.fees, 2))
        self.transaction['gain'].append(round(self.gain, 2))
        self.transaction['profit'].append(round(self.profit, 2))
        self.transaction['capital'].append(round(self.capital, 2))


    def reset_sell(self):

        """ reset the variables for a sell occurred """

        self.sell_price = 0.
        self.fees = 0.

        self.profit -= self.gain
        self.capital -= self.gain
        self.gain = 0


    def reset(self):

        """ reset the trading variables after a transaction """

        self.coin = ''
        self.buy_price = 0.
        self.sell_price = 0.
        self.stoploss_price = 0.
        self.nb_coins = 0.
        self.fees = 0.
        self.gain = 0.
        self.status = 'free'


    def full_reset(self):

        """ full resent of all the agent's variables """

        # trading parameters
        self.capital = self.deposited

        # trading variables
        self.status = 'free'
        self.coin = ''
        self.buy_price = 0.
        self.sell_price = 0.
        self.stoploss_price = 0.
        self.watermark = 0.
        self.expected_profit = 0.
        self.nb_coins = 0.
        self.fees = 0.
        self.gain = 0.
        self.profit = 0.

        self.count = 0
        self.transaction = {'trade': [],
                            'time': [],
                            'coin': [],
                            'buy': [],
                            'sell': [],
                            'stop': [],
                            'fees': [],
                            'gain': [],
                            'profit': [],
                            'capital': []}

    def get_record(self):
        return self.transaction

    def get_capital(self):
        return self.capital

    def get_profit(self):
        return self.profit

    def get_count(self):
        return self.count

    def get_status(self):
        return self.status

    def get_deposited(self):
        return self.deposited




class Env:

    """
    The environment in which the trading happen
    """

    def __init__(self, capital: int, a_k: str, a_s: str,
                 dt: str, t0: str, ulogs: int, fake=False, env='win'):

        """
        :param capital: amount of money invested
        :param a_k: public key
        :param a_s: secret key
        :param dt: candlestick timestep: ('1m', '3m', '5m', '15m', '30m', '1h', '4h', '8h', '1d', '1w')
        :param t0: period considered: ('1 hours', '2 hours', '8 hours', '1 days', 'n* days')
        :param ulogs: log update timestep in minutes (int)
        :param coins:
        :param fake: fake run or not (True or False)
        :param env: OS environment ('win' for Windows or 'linux')
        """

        # trader
        self.trader = Trader(capital=capital, fees_percent=0.00075)
        self.fake = fake


        # name
        self.name = ''
        for _ in range(2):
            self.name += np.random.choice(('au', 'ken', 'po', 'li', 'se', 'ry', 'gu', 'chi', 'we', 'tu', 'fra', 'qu',
                                           'bi', 'tri', 'quadri', 'lu', 'mo', 'no', 'xi', 'zen', 'hun'))
        self.name += np.random.choice(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

        # parameters
        self.a_k = a_k
        self.a_s = a_s
        self.exchange = ccxt.binance({'apiKey': a_k, 'apiSecret': a_s, 'enableRateLimit': True})
        self.client = Client(a_k, a_s)
        self.interval = Client.KLINE_INTERVAL_1MINUTE
        self.nb_candles = 100

        self.nb_coins = 10  # number of coins selected for a given run
        self.selected_coins = []

        # coin pool: feel free to select the coins you like
        self.coins = ('1INCH', 'AAVE', 'ACM', 'ADA', 'AGLD', 'AION', 'AKRO', 'ALGO',
                      'ALICE', 'ALPACA', 'ALPHA', 'ANKR', 'ANT', 'ARDR', 'ARPA', 'ASR',
                      'ATA', 'ATM', 'ATOM', 'AUD', 'AUDIO', 'AUTO', 'AVA', 'AVAX', 'AXS',
                      'BADGER', 'BAKE', 'BAL', 'BAND', 'BAR', 'BAT', 'BCH',
                      'BEAM', 'BEL', 'BETA', 'BLZ', 'BNB', 'BNT', 'BOND', 'BTC',
                      'BTCST', 'BTG', 'BTS', 'BTT', 'BURGER', 'BUSD', 'BZRX', 'C98',
                      'CAKE', 'CELO', 'CELR', 'CFX', 'CHR', 'CHZ', 'CKB', 'CLV', 'COCOS',
                      'COMP', 'COS', 'COTI', 'CRV', 'CTK', 'CTSI', 'CTXC', 'CVC', 'CVP',
                      'DASH', 'DATA', 'DCR', 'DEGO', 'DENT', 'DEXE', 'DGB', 'DIA',
                      'DNT', 'DOCK', 'DODO', 'DOGE', 'DOT', 'DREP', 'DUSK', 'DYDX',
                      'EGLD', 'ELF', 'ENJ', 'EOS', 'EPS', 'ERN', 'ETC', 'ETH',
                      'EUR', 'FARM', 'FET', 'FIDA', 'FIL', 'FIO', 'FIRO', 'FIS', 'FLM',
                      'FLOW', 'FOR', 'FORTH', 'FRONT', 'FTM', 'FTT', 'FUN', 'GALA',
                      'GBP', 'GHST', 'GNO', 'GRT', 'GTC', 'GTO', 'GXS', 'HARD', 'HBAR',
                      'HIVE', 'HNT', 'HOT', 'ICP', 'ICX', 'IDEX', 'ILV', 'INJ',
                      'IOST', 'IOTA', 'IOTX', 'IRIS', 'JST', 'JUV', 'KAVA', 'KEEP',
                      'KEY', 'KLAY', 'KMD', 'KNC', 'KSM', 'LINA', 'LINK', 'LIT',
                      'LPT', 'LRC', 'LSK', 'LTC', 'LTO', 'LUNA', 'MANA', 'MASK', 'MATIC',
                      'MBL', 'MBOX', 'MDT', 'MDX', 'MFT', 'MINA', 'MIR', 'MITH',
                      'MKR', 'MLN', 'MTL', 'NANO', 'NBS', 'NEAR', 'NEO', 'NKN', 'NMR',
                      'NU', 'NULS', 'OCEAN', 'OGN', 'OMG', 'ONE', 'ONG', 'ONT',
                      'ORN', 'OXT', 'PAXG', 'PERL', 'PERP', 'PHA', 'PNT', 'POLS',
                      'POLY', 'POND', 'PSG', 'PUNDIX', 'QNT', 'QTUM', 'QUICK', 'RAD',
                      'RAMP', 'RAY', 'REEF', 'REN', 'REP', 'REQ', 'RIF', 'RLC', 'ROSE',
                      'RSR', 'RUNE', 'RVN', 'SAND', 'SFP', 'SHIB', 'SKL', 'SLP', 'SNX',
                      'SOL', 'SRM', 'STMX', 'STORJ', 'STPT', 'STRAX', 'STX',
                      'SUN', 'SUPER', 'SUSD', 'SUSHI', 'SXP', 'SYS', 'TCT', 'TFUEL',
                      'THETA', 'TKO', 'TLM', 'TOMO', 'TORN', 'TRB', 'TRIBE', 'TROY',
                      'TRU', 'TRX', 'TVK', 'TWT', 'UMA', 'UNFI', 'UNI', 'USDP', 'UTK',
                      'VET', 'VIDT', 'VITE', 'VTHO', 'WAN', 'WAVES', 'WAXP',
                      'WIN', 'WING', 'WNXM', 'WRX', 'WTC', 'XEC', 'XEM', 'XLM', 'XMR',
                      'XRP', 'XTZ', 'XVG', 'XVS', 'YFI', 'YFII', 'YGG', 'ZEC', 'ZEN',
                      'ZIL', 'ZRX')


        self.prec = ()
        self.all_coins = self.coins
        self.update_cache = 10
        self.update_logs = ulogs * 60

        # data variable
        self.N_coins = len(self.coins)
        self.idx = 0
        self.dt = dt
        self.t0 = t0
        self.duration = 50
        self.dt_ = int(self.dt[:-1])
        self.t0 = self.t0
        self.period = int(60 * 24 / self.dt_) * int(self.t0[:-5])
        self.data_coin = {}
        self.final_data = []

        # variables
        self.coin = ''

        # record
        self.start_time = {}
        self.start_month = time.localtime().tm_mon
        self.start_day = time.localtime().tm_mday
        self.start_hour = time.localtime().tm_hour
        self.start_min = time.localtime().tm_min

        # type of run
        if not fake:

            # build directory
            self.env = env
            if env == 'win':
                self.botpath = os.getcwd() + '\\sessions'
            else:
                self.botpath = os.getcwd() + '//sessions'

            os.mkdir(self.botdir)
            os.chdir(self.botdir)

            print('\n<session directory built>')
            print('\n@real run!')

        else:
            print('\n@fake run!')

        print('\n<Environment built>')


    def get_next_coin(self):

        """ return the next coin """

        self.coin = np.random.choice(self.selected_coins)

        return self.coin

    def get_next_hlcv(self, same=False):

        """ return the next/same coin's hlc """

        time.sleep(0.01)

        if not same:
            self.coin = self.get_next_coin()

        try:
            package = np.array(self.client.get_klines(symbol=self.coin + 'USDT',
                                                      interval=self.interval, limit=self.nb_candles)).astype(float)

            return (package[:, 2], package[:, 3], package[:, 4], package[:, 5]), package[-1, 4]


        except (binance.exceptions.BinanceAPIException, requests.exceptions.ConnectionError) as e:
            if e.errno != -1121:
                print('looked for ', self.coin)

            return self.get_next_hlc(same=same)


        except (binance.exceptions.BinanceAPIException, requests.exceptions.ConnectionError) as e:
            if e.errno != 10054:
                time.sleep(1)
                return self.get_next_hlc(same=same)

            self.reconnect()

            if self.env == 'win':
                os.system('cls')
            else:
                os.system('clear')
            print()
            print(' ----------------------')
            print(' | something is wrong |')
            print(' ----------------------')
            print()
            print(' retrying connecting in 5s ...')
            time.sleep(5)
            return


        except ConnectionError:
            if self.env == 'win':
                os.system('cls')
            else:
                os.system('clear')
            print()
            print(' ----------------------------')
            print(' | Binance is not reachable |')
            print(' ----------------------------')
            print()
            print(' retrying connecting in 5s ...')
            time.sleep(5)
            return self.get_next_hlc(same=same)

        except:
            if self.env == 'win':
                os.system('cls')
            else:
                os.system('clear')
            print()
            print(' ---------------------------')
            print(' | Binance is not reachable|')
            print(' ---------------------------')
            print('\n something weird happened')
            print()
            print(' retrying connecting in 5s ...')
            time.sleep(5)

            return self.get_next_hlc(same=same)

    def reconnect(self):

        """ if the connections is temporarily disrupted """

        self.client = Client(self.a_k, self.a_s)

    def place_market_buy(self, quantity):

        """ execute a market buy order """

        try:
            self.client.order_market_buy(symbol=self.coin + 'USDT', quantity=quantity)
            print(f'\n@bought {self.coin} <---')
            return True

        except:
            self.client.order_market_buy(symbol=self.coin + 'USDT', quantity=quantity)

            print('!! Something weird happened while buying')
            return False

    def place_market_sell(self, quantity):

        """ execute a market sell order """

        try:
            self.client.order_market_sell(symbol=self.coin + 'USDT', quantity=quantity)
            print(f'@sold {self.coin} --->')
            return True

        except:
            print('!! Something weird happened while selling')
            return False

    def save_logs(self):

        """ save logs """

        # history
        with open('record.json', 'w') as rec:
            rec.write(json.dumps(self.trader.get_record()))

        # report
        report = f'\n-- Trading Bot Report @real --\n'

        date = time.localtime()
        if not self.fake:
            report += '\nStart: {} at {}\nUpdate: {}/{}/{} at {}:{}:{}\n'.format(
                self.start_time['days'], self.start_time['hours'],
                date.tm_mday, date.tm_mon, date.tm_year, min(date.tm_hour, 24), date.tm_min, date.tm_sec
            )
        report += '-' * 33
        report += f'\n updates: {self.update_logs}'
        report += f'\n trades:   {self.trader.get_count()}'
        report += f'\n deposit:  {self.trader.get_deposited()}$'
        report += f'\n capital:    {self.trader.get_capital():.3f}$'
        report += f'\n profit:   {self.trader.get_profit():.3f}$\n'
        if self.trader.get_profit() > 0:
            report += '\n net: POSITIVE'
        else:
            report += '\n net: NEGATIVE'
        report += f'\n'

        with open('report.txt', 'w') as f:
            f.write(report)


    def get_data(self):

        """ collect coin data """

        for _ in range(self.nb_coins):
            coin = np.random.choice(self.coins)
            self.selected_coins.append(coin)
            self.data_coin[coin] = []

        print('\nselected coins: ', list(self.selected_coins))
        print(f'\n{self.dt} candlesticks for {self.t0}')

        self.duration = 40
        damaged = []
        highs, lows, closes, vols = 0., 0., 0., 0.

        print('\nloaded:')
        for cidx, coin in enumerate(self.selected_coins):
            values = self.client.get_historical_klines(symbol=coin + 'USDT',
                                                       interval=self.dt,
                                                       start_str=self.t0 + ' ago UTC')

            values = np.array(values).astype(float)

            try:
                highs, lows, closes, vols = values[:, 1], values[:, 3], values[:, 4], values[:, 5]
            except IndexError:
                print(values.shape)
                print('damaged coin ', coin)
                damaged.append(cidx)

            for t in range(self.duration, self.period):
                self.data_coin[coin].append([highs[t - self.duration: t], lows[t - self.duration: t],
                                             closes[t - self.duration: t], vols[t - self.duration: t]])

            print(cidx + 1, ' ', coin)

        for cx in damaged:
            del self.coins[cx]
            self.N_coins -= 1

        self.final_data = [[self.data_coin[coin][t] for coin in self.selected_coins] for t in range(self.period - self.duration)]

        print('\n<all data collected>')
        print()
        time.sleep(1)

    def set_step(self):

        """ define the stepsize and minimum price for an order """

        idx = 0

        pack = self.client.get_symbol_info(self.coin + 'USDT')
        for idx, i in enumerate(pack['filters'][0]['tickSize'][2:]):
            if i == '1':
                break

        return int(idx) + 1, float(pack['filters'][0]['minPrice'])

    def round_down(self, coin, number):

        """ get the coin quantity rounded down """

        info = self.client.get_symbol_info('%sUSDT' % coin)
        step_size = [float(_['stepSize']) for _ in info['filters'] if _['filterType'] == 'LOT_SIZE'][0]
        step_size = '%.8f' % step_size
        step_size = step_size.rstrip('0')
        decimals = len(step_size.split('.')[1])
        return np.floor(number * 10 ** decimals) / 10 ** decimals

    def main(self):

        """ run with real money: an agent will check and complete buy/sell orders given a pre-defined algorithm """

        clock = time.localtime()
        self.start_time['days'] = f'{clock.tm_mday}/{clock.tm_mon}/{clock.tm_year}'
        self.start_time['hours'] = f'{min(clock.tm_hour, 24)}:{clock.tm_min}:{clock.tm_sec}'

        start = time.time()

        self.save_logs()

        print()
        print('-' * 50)
        print('\nrunning...')

        while True:

            # wait for buying
            if self.trader.get_status() == 'free':

                hlc, price = self.get_next_hlcv()

                # check buy conditions
                if self.trader.check_control(hlcv=hlcv, kind='buy'):

                    # actually buying
                    quantity = self.trader.buy(price=price, coin=self.coin)
                    stepsize, minqty = self.set_step()
                    quantity = self.round_down(self.coin, quantity)

                    # check quantity
                    if quantity < minqty:
                        continue

                    # run order
                    if not self.place_market_buy(quantity=quantity):
                        self.trader.reset()  # something wrong
                        continue

                    self.chat.register_coin(self.coin)

            # wait for selling
            else:

                hlcv, price = self.get_next_hlcv(same=True)

                # check sell conditions
                if self.trader.check_control(hlcv=hlcv, kind='sell'):

                    # actually selling
                    _ = self.trader.sell(price=price)

                    # check quantity
                    if quantity < minqty:
                        print('?? the quantity has changed? (', quantity, ')')
                        continue

                    # run order
                    if not self.place_market_sell(quantity=quantity):
                        self.trader.reset_sell()  # something wrong
                        continue
                    else:
                        self.trader.record()
                        self.trader.reset()
                        self.chat.delete_coin(self.coin, self.trader.get_profit())

            # record
            if (time.time() - start) > self.update_logs:
                self.save_logs()
                start = time.time()
                time.sleep(1)

        self.save_logs()

        print('\n<Session Terminated> >logs saved< ')


    def get_next_fake_hlcv(self, t: int, same=False):

        """ return next indexed hlc """

        if not same:
            self.idx = np.random.randint(0, self.nb_coins - 1)
            self.coin = self.selected_coins[self.idx]

            return self.final_data[t][self.idx], self.final_data[t][self.idx][2][-1]
        else:
            return self.final_data[t][self.idx], self.final_data[t][self.idx][2][-1]

    def avg_fake(self, N: int):

        """ average over N fake run """

        # collect data
        self.get_data()

        profits = []
        transactions = {}

        print('\nrunning...\n')

        for session in range(N):

            # if session % 10 == 0:
            #    self.get_data()

            if session % 25 == 0:
                print(f'\nSession {session + 1}|{N}')

            profit, transactions = self.main_fake()
            self.trader.full_reset()
            profits.append(profit)

        print('\n<end>')

        mu, sigma = sum(profits) / N, np.std(profits)

        if sigma == 0:
            surv_perc = 100
        else:
            surv_perc = min((scipy.stats.norm.cdf(abs(-mu / sigma)) * 100, 100))

        print()
        print('-'*50)
        print(f'\nAverage profit for {self.dt}/{self.t0} in {N} sessions: {mu:.2f}$ (std: {sigma:.3f}$)')
        print(f'P-value of positive profit: {surv_perc:.1f}%')   # chances of obtaining a positive profit
        print()
        print("last agent's transaction sheet:")
        print(pd.DataFrame(transactions))

    def main_fake(self):

        """ fake run: given a trading algorithm, an agent is set up and let run over historic data """

        # collect data
        start = time.time()

        for t in range(self.period - self.duration):

            # wait for buying
            if self.trader.get_status() == 'free':

                hlcv, price = self.get_next_fake_hlc(t)

                # check buy conditions
                if self.trader.check_control(hlcv=hlcv, kind='buy'):

                    # buying
                    _ = self.trader.buy(price=hlcv[2][-1], coin=self.coin)


            # wait for selling
            else:

                hlcv, price = self.get_next_fake_hlcv(t=t, same=True)

                # check sell conditions
                if self.trader.check_control(hlcv=hlcv, kind='sell'):

                    # selling
                    _ = self.trader.sell(price=price)
                    self.trader.record(timestamp=t)
                    self.trader.reset()

            # record
            if (time.time() - start) > self.update_logs:
                self.save_logs()
                start = time.time()
                time.sleep(1)

        return self.trader.get_profit(), self.trader.transaction

