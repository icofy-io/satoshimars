
from datetime import timedelta, datetime, timezone
import time
import operator
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
import pandas as pd
from threading import Thread
from model_ta import EnsembleModel
import json
# Statistical Arbitrage
# Mark Price is Avg Price of : Bitmex, Coinbase and Bitfinex
# Last Price is simply Bitmex Spot Price


from bitmex import bitmex

ID = 'GUMQ-8G_QhurgblQiQSBG_MA'
SECRET = 'OHKwTaD4FxuvS-s0qIJmO5Q8vrdXMT5AX5EQ76r5F4s4SbeZ'


# classes provide a means of bundling data and functionality together.
# creating a new class creates a new type of object, allowing new instances of that type to be made.
# each class instance can have attributes attached to it for maintaining its state.
# class instances can also have methods (defined by its class) for modifying its state.
class BitmexObject:  # this is the class variable that will be shared by all instances

    def __init__(self,
                 attr_dict):  # When a class defines an __init__() method, class instantiation automatically invokes __init__() for the newly-created class instance.
        self.__dict__.update(
            attr_dict)  # self.__dict__ is an instance variable unique to each instance, attr_dict is attribute dictionary that allows its elements to be accessed both as keys and as attributes


# now, we are utilizing our newly defined class BitmexObject for 3 instances: Being Order(BitmexObject), Position(BitmexObject), and InstrumentQuery(BitmexObject):

class Order(BitmexObject):

    def __repr__(
            self):  # object.__repr__(self): called by the repr() built-in function and by string conversions (reverse quotes) to compute the "official" string representation of an object
        return f'{self.side}({self.symbol}, {self.orderQty})'  # define what we want our defined representation for our object to return


class Position(BitmexObject):

    def __repr__(
            self):  # object.__repr__(self): called by the repr() built-in function and by string conversions (reverse quotes) to compute the "official" string representation of an object
        return f'Position({self.symbol}, {self.currentQty})'  # define what we want our defined representation for our object to return


class InstrumentQuery(BitmexObject):

    def __repr__(
            self):  # object.__repr__(self): called by the repr() built-in function and by string conversions (reverse quotes) to compute the "official" string representation of an object
        return f'InstrumentQuery({self.symbol}, {self.lastPrice}, {self.markPrice})'  # define what we want our defined representation for our object to return


# class Bitmex, ***** note: no longer utilizing BitmexObject though on class Bitmex:


class Bitmex:  # define new class called Bitmex , which defines algorithm framework and parameters with different methods

    def __init__(self, api_key, api_secret,
                 test=True):  # first, need to define our _init_ instance before our algo methods
        self._client = bitmex(  # where we define bitmex as client for algorithm
            test=test,
            api_key=api_key,
            api_secret=api_secret
        )
        self._orders = {}
        self.stop = None
        self.target = None

    # NOTE: Only the first 3 methods (get_intrument_query, _enter_market, and get_positions required us to set up class Bitmex objects above before we defined class Bitmex.
    # The last methods in class Bitmex (buy, sell place_stop, place_target, calc_targets_ and enter_bracket did not require this.

    def get_instrument_query(self, symbol):  # retrieve prices for mark and last price
        json, adapter = self._client.Instrument.Instrument_get(symbol=symbol, reverse=True, count=1).result()
        inst = InstrumentQuery(json[0])  # InstrumentQuery is class we have defined above
        print(f'{symbol}:, Last: {inst.lastPrice}, Mark: {inst.markPrice}')
        return inst

    def get_dataframe(self, symbol, timeframe):
        tf_deltas = {
            '1m': timedelta(minutes=750),
            '5m': timedelta(minutes=750 * 5),
            '1h': timedelta(hours=750),
            '1d': timedelta(days=750),
        }
        start = datetime.now(timezone.utc) - tf_deltas[timeframe]
        json, adapter = self._client.Trade.Trade_getBucketed(
            binSize=timeframe,
            symbol=symbol,
            count=750,
            startTime=start,
            columns='open, high, low, close'
        ).result()
        return pd.DataFrame.from_records(
            data=json,
            columns=['timestamp', 'open', 'high', 'low', 'close'],
            index='timestamp',

        )

    def _enter_market(self, symbol, size,
                      **kws):  # **kwargs allows you to pass keyworded variable length of arguments to a function
        order = self._client.Order.Order_new(symbol=symbol, orderQty=size, **kws)
        json, adapter = order.result()
        order = Order(json)  # Order is class we have defined above
        self._orders[order.orderID] = order
        return order

    def get_positions(self):
        positions, adapter = self._client.Position.Position_get().result()
        return [Position(p) for p in positions if p["currentQty"] != 0]  # Position is class we have defined above

    def buy(self, symbol, size, **kws):
        return self._enter_market(symbol, abs(size),
                                  **kws)  # **kwargs allows you to pass keyworded variable length of arguments to a function

    def sell(self, symbol, size, **kws):
        return self._enter_market(symbol, -1 * abs(size),
                                  **kws)  # -1 for sell,  **kwargs allows you to pass keyworded variable length of arguments to a function

    def place_stop(self, symbol, size, price, *, side):
        func = getattr(self, side)
        return func(
            symbol,
            size,
            stopPx=price,  # stopPx = price for stop
        )

    def place_target(self, symbol, size, price, *, side):
        func = getattr(self, side)
        return func(
            symbol,
            size,
            price=price,  # standard price = price for limit
        )

    def _calc_targets(self, entry_price, side, target_offset=20,
                      stop_offset=40):  # calculate targets for stops and limits
        if side == 'buy':
            stop_op = operator.sub
            target_op = operator.add
        else:
            stop_op = operator.add
            target_op = operator.sub
        stop_price = stop_op(entry_price, stop_offset)
        # stop_price = round(stop_price, 0.5)
        target_price = target_op(entry_price, target_offset)
        # target_price = round(target_price, 0.5)
        return stop_price, target_price

    def enter_bracket(self, symbol, size, side='buy', target_offset=20, stop_offset=40):  # create bracket order
        assert side in {'buy', 'sell'}, 'Side must be buy or sell'
        exit_side_map = {'buy': 'sell', 'sell': 'buy'}
        print(f'Submitting Market order for {symbol}')
        entry = getattr(self, side)(symbol, size)
        if entry.ordStatus == 'Filled':
            link_id = entry.orderID
            exit_side = exit_side_map[side]
            stop_price, target_price = self._calc_targets(
                entry.avgPx, side, target_offset, stop_offset)
            self.stop = self.place_stop(symbol, size, stop_price, side=exit_side)
            self.target = self.place_target(symbol, size, target_price, side=exit_side)

    def check_exit_fill(self):
        json_, adapter = self._client.Order.Order_get(filter=json.dumps({"open": True})).result()
        orders = [Order(j) for j in json_]
        if self.stop and self.target and len(orders) == 1:
            order_id = orders[0].orderID
            print('One live order left, should be a stop or a target')
            assert order_id in {self.stop.orderID, self.target.orderID}
            print(f'Sending cancelation request for {orders[0]}')
            self._client.Order.Order_delete(orderID=order_id)
            print('Order should be canceled now, resetting target and stop attrs to None')
            self.stop = None
            self.target = None


# Trader Class, ***** note: no longer utilizing class BitmexObject though on any methods

class Trader:  # define another new class called Trader, where trading happens

    """""""""
    Available Bitmex timeframe arguments:

    tf_deltas = {
        '1m': timedelta(minutes=750),
        '5m': timedelta(minutes=750 * 5),
        '1h': timedelta(hours=750),
        '1d': timedelta(days=750),
    }
    """""""""""

    def __init__(self, symbol, model, size, timeframe='5m',
                 # again, first, need to define our _init_ instance before our algo methods
                 target_offset=20, stop_offset=40):
        self.symbol = symbol
        self.model = model
        self.size = size
        self.timeframe = timeframe
        self.target_offset = target_offset
        self.stop_offset = stop_offset
        self.bit = Bitmex(ID, SECRET)  # for live trading, change to self.bit = Bitmex(ID, SECRET, test=False)

    def _run(self, func, sleep_time):
        while self.live:
            try:
                func()
            except Exception as ex:
                print(f'Error running {func.__name__}:', ex)
                break
            time.sleep(sleep_time)

    def run(self):
        self.live = True
        exit_thread = Thread(target=self.exit_runner)
        exit_thread.start()
        self.entry_runner()
        exit_thread.join()

    def entry_runner(self):
        self._run(self._enter_if_flat_and_good_price, 30)

    def exit_runner(self):
        self._run(self.bit.check_exit_fill, 5) # update every 5 seconds

    def _trade_criteria(self):
        positions = self.bit.get_positions()
        return len(positions) == 0

    def _predict(self):
        df = self.bit.get_dataframe(self.symbol, self.timeframe)
        print(f'Got data on {self.symbol} from {df.index[0]} to {df.index[-1]}')
        predict = self.model.run(df)
        print('Prediction: ', predict)
        return predict

    def _enter_if_flat_and_good_price(self):
        if self._trade_criteria():
            print('No positions, will run prediction and enter market')
            signal = self._predict()
            if signal == 1:
                side = 'buy'
            elif signal == -1:
                side = 'sell'
            elif signal == 0:
                print('Signal is 0, staying flat')
                return
            else:
                raise ValueError(f'Unexpected signal: {signal}, should have been 0, 1 or -1')
            self.bit.enter_bracket(
                self.symbol,
                self.size,
                side=side,
                target_offset=self.target_offset,
                stop_offset=self.stop_offset
            )
        else:
            print('Already in a trade and not entering')


if __name__ == '__main__':  # where we call our Trader class defined above, in main
    model = EnsembleModel()
    trader = Trader('XBTUSD', model, 450, '1h',
                    target_offset=30, stop_offset=50)
    try:
        trader.run()
    except:
        trader.live = True
