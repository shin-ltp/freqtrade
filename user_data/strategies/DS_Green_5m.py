# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, BooleanParameter, DecimalParameter, IntParameter, CategoricalParameter
import math
import logging

logger = logging.getLogger(__name__)
########################################################################################################################################################
# EWO
########################################################################################################################################################
def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif
########################################################################################################################################################
# pct_change
########################################################################################################################################################   
def pct_change(a, b):
    return (b - a) / a
########################################################################################################################################################
class DS_Green_5m(IStrategy):
########################################################################################################################################################
# Hyperopt
########################################################################################################################################################
    buy_params = {
        "lambo2_enabled": True,
        "ewo1_enabled": True,
        "ewo2_enabled": True,
        "cofi_enabled": True,
        # Ewo
        "base_nb_candles_buy": 16,
        # EWO 1
        "ewo_high": 3.2,
        "low_offset_1": 0.998,
        "high_offset_1": 0.994,
        "rsi_buy": 42,
        # Ewo 2
        "ewo_low": -6.0,
        "low_offset_2": 0.996,
        "high_offset_2": 1.012,
        # Lambo 2
        "lambo2_ema_14_factor": 0.997,
        "lambo2_rsi_14_limit": 42,
        "lambo2_rsi_4_limit": 36,
        # Cofi
        "buy_adx": 40,
        "buy_fastd": 30,
        "buy_fastk": 30,
        "buy_ema_cofi": 0.994,
        "buy_ewo_high": 3.0,
        # DCA
        "dca_min_rsi": 55,
        "max_safety_orders": 2,
        "initial_safety_order_trigger": -0.01,
    }
    sell_params = {
        "base_nb_candles_sell": 16,
        # Ewo
        "high_offset_above": 1.012,
        "high_offset_below": 1.015,
        # custom stoploss params
        "pHSL": -0.10,
        "pPF_1": 0.03,
        "pPF_2": 0.10,
        "pSL_1": 0.02,
        "pSL_2": 0.04,
        # unclog exit params
        "unclog_threshold": 0.012,
        "unclog_wait_time": 90,
        "unclog_rsi_threshold": 48,
        "unclog_profit_ratio": 0.008,
    }
    minimal_roi = {
        "0": 0.10,
        "10": 0.07,
        "20": 0.05,
        "40": 0.03
    }
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.035
    trailing_only_offset_is_reached = True
########################################################################################################################################################
# Main
########################################################################################################################################################
    use_custom_stoploss = True
    timeframe = '5m'
    inf_1h = '1h'
    # Sell signal configuration.
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.015
    ignore_roi_if_entry_signal = True
    # Number of past candles to consider upon startup.
    process_only_new_candles = True
    startup_candle_count = 100
    # Adjsut trade position
    safety_order_step_scale = 1.2
    safety_order_volume_scale = 1.3
    # Configuration of order types.
    order_types = {
        'entry': 'limit',
        'exit': 'market',
        'trailing_stop_loss': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }
    # Plotting configuration for visualizing indicators in backtesting.
    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},  # Color for the buy moving average.
            'ma_sell': {'color': 'orange'},  # Color for the sell moving average.
        },
    }
    # Order Time-In-Force defines how long an order will remain active before it is executed or expired.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }
########################################################################################################################################################
# Parameters
########################################################################################################################################################
    # Enabled
    is_optimize_remove = False
    lambo2_enabled = BooleanParameter(default=buy_params['lambo2_enabled'], space='buy', optimize=is_optimize_remove)
    ewo1_enabled = BooleanParameter(default=buy_params['ewo1_enabled'], space='buy', optimize=is_optimize_remove)
    ewo2_enabled = BooleanParameter(default=buy_params['ewo2_enabled'], space='buy', optimize=is_optimize_remove)
    cofi_enabled = BooleanParameter(default=buy_params['cofi_enabled'], space='buy', optimize=is_optimize_remove)
############################################################################################################################################################################
    # Candles
    is_optimize_base_nb_candles = True
    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=is_optimize_base_nb_candles)
    # EWO Protection
    fast_ewo = 60
    slow_ewo = 220
    # EWO 1 
    is_optimize_ewo = False # Optimized 7/11/24 for 2 days
    low_offset_1 = DecimalParameter(0.985, 0.995, default=buy_params['low_offset_1'], space='buy', optimize=is_optimize_ewo)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=is_optimize_ewo)
    ewo_high = DecimalParameter(3.0, 3.4, default=buy_params['ewo_high'], space='buy', optimize=is_optimize_ewo)
    high_offset_1 = DecimalParameter(0.95, 1.10, default=buy_params['high_offset_1'], space='buy', optimize=is_optimize_ewo)
    # Ewo 2
    is_optimize_ewo2 = False # Optimized 8/11/24 for 6 h
    ewo_low = DecimalParameter(-20.0, -8.0,default=buy_params['ewo_low'], space='buy', optimize=is_optimize_ewo2)
    low_offset_2 = DecimalParameter(0.985, 0.995, default=buy_params['low_offset_2'], space='buy', optimize=is_optimize_ewo2)
    high_offset_2 = DecimalParameter(0.95, 1.10, default=buy_params['high_offset_2'], space='buy', optimize=is_optimize_ewo2)
    # lambo2
    is_optimize_lambo2 = False # Optimized 10/11/24 for 6 h
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3,  default=buy_params['lambo2_ema_14_factor'], space='buy', optimize=is_optimize_lambo2)
    lambo2_rsi_4_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=is_optimize_lambo2)
    lambo2_rsi_14_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=is_optimize_lambo2)
    #cofi
    is_optimize_cofi = False # Optimized 8/11/24 for 4h
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97, space='buy', optimize = is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, space='buy', optimize = is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, space='buy', optimize = is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, space='buy', optimize = is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, space='buy', default=3.553, optimize = is_optimize_cofi)
    # ?
    dca_min_rsi = IntParameter(35, 75, default=buy_params['dca_min_rsi'], space='buy', optimize=False)
    # Sell
    is_optimize_offset_sell = False # Optimized 9/11/24 for 13h
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=is_optimize_offset_sell)
    high_offset_above = DecimalParameter(1.00, 1.10, default=sell_params['high_offset_above'], space='sell', optimize=is_optimize_offset_sell)
    high_offset_below = DecimalParameter(0.95, 1.05, default=sell_params['high_offset_below'], space='sell', optimize=is_optimize_offset_sell)
############################################################################################################################################################################
    # Custom Stoploss
    is_optimize_stoploss = False # Optimized 7/11/24
    # Hard stoploss profit
    pHSL = DecimalParameter(-0.500, -0.040, default=-0.08, decimals=3, space='sell', optimize=is_optimize_stoploss, load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', optimize=is_optimize_stoploss, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', optimize=is_optimize_stoploss, load=True)
    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell',optimize=is_optimize_stoploss, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', optimize=is_optimize_stoploss,load=True)
########################################################################################################################################################
# Informative Pairs
########################################################################################################################################################
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        informative_pairs.append((btc_info_pair, self.timeframe))
        informative_pairs.append((btc_info_pair, self.inf_1h))

        return informative_pairs
########################################################################################################################################################
# Informative Pairs
########################################################################################################################################################
    def base_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['price_trend_long'] = (dataframe['close'].rolling(8).mean() / dataframe['close'].shift(8).rolling(144).mean())
        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        
        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)
        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe
########################################################################################################################################################
# Indicators
########################################################################################################################################################
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 计算ma_sell指标
        dataframe['ma_sell'] = ta.SMA(dataframe['close'], timeperiod=20)
        
        if self.config['stake_currency'] in ['USDT','BUSD']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        btc_info_tf = self.dp.get_pair_dataframe(btc_info_pair, self.inf_1h)
        btc_info_tf = self.info_tf_btc_indicators(btc_info_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_info_tf, self.timeframe, self.inf_1h, ffill=True)
        drop_columns = [f"{s}_{self.inf_1h}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        btc_base_tf = self.dp.get_pair_dataframe(btc_info_pair, self.timeframe)
        btc_base_tf = self.base_tf_btc_indicators(btc_base_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_base_tf, self.timeframe, self.timeframe, ffill=True)
        drop_columns = [f"{s}_{self.timeframe}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        #lambo2
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        # Pump strength
        dataframe['dema_30'] = ta.DEMA(dataframe, period=30)
        dataframe['dema_200'] = ta.DEMA(dataframe, period=200)
        dataframe['pump_strength'] = (dataframe['dema_30'] - dataframe['dema_200']) / dataframe['dema_30']
        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        # Dump Protection
        dataframe = self.pump_dump_protection(dataframe, metadata)
        # RSI
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        return dataframe
########################################################################################################################################################
# Pump Dump Protection
########################################################################################################################################################
    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    
        df36h = dataframe.copy().shift( 432 ) # TODO FIXME: This assumes 5m timeframe
        df24h = dataframe.copy().shift( 288 ) # TODO FIXME: This assumes 5m timeframe
        
        dataframe['volume_mean_short'] = dataframe['volume'].rolling(4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
        dataframe['volume_mean_base'] = df36h['volume'].rolling(288).mean()
        
        dataframe['volume_change_percentage'] = (dataframe['volume_mean_long'] / dataframe['volume_mean_base'])
        dataframe['rsi_mean'] = dataframe['rsi'].rolling(48).mean()
        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0), -1, 0)
        
        return dataframe
########################################################################################################################################################
# Custom Stoploss
########################################################################################################################################################
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value
        
        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PF_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.
        
        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL
    
        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99
    
        return stoploss_from_open(sl_profit, current_profit)
########################################################################################################################################################
# Buy Trend
########################################################################################################################################################
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        dataframe.loc[:, 'enter_long'] = 0
    
        # Lambo2 condition
        lambo2 = (
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))
        )
        dataframe.loc[lambo2, 'enter_tag'] += 'lambo2_'
        conditions.append(lambo2)
    
        # Buy1 EWO condition
        ewo = (
            (dataframe['rsi_fast'] < 35) &
            (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_1.value)) &
            (dataframe['EWO'] > self.ewo_high.value) &
            (dataframe['rsi'] < self.rsi_buy.value) &
            (dataframe['volume'] > 0) &
            (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_1.value))
        )
        dataframe.loc[ewo, 'enter_tag'] += 'eworsi_'
        conditions.append(ewo)
    
        # Buy2 EWO condition
        ewo2 = (
            (dataframe['rsi_fast'] < 35) &
            (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
            (dataframe['EWO'] < self.ewo_low.value) &
            (dataframe['volume'] > 0) &
            (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value))
        )
        dataframe.loc[ewo2, 'enter_tag'] += 'ewo2_'
        conditions.append(ewo2)
    
        # COFI condition
        cofi = (
            (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
            (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
            (dataframe['fastk'] < self.buy_fastk.value) &
            (dataframe['fastd'] < self.buy_fastd.value) &
            (dataframe['adx'] > self.buy_adx.value) &
            (dataframe['EWO'] > self.buy_ewo_high.value)
        )
        dataframe.loc[cofi, 'enter_tag'] += 'cofi_'
        conditions.append(cofi)
    
        # Applying buy conditions
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_long'] = 1
    
        # Additional conditions to avoid buying
        dont_buy_conditions = []
    
        # don't buy if there seems to be a Pump and Dump event.
        dont_buy_conditions.append((dataframe['pnd_volume_warn'] < 0.0))
    
        # BTC price protection
        dont_buy_conditions.append((dataframe['btc_rsi_8_1h'] < 35.0))
    
        # Applying don't buy conditions
        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'enter_long'] = 0
    
        return dataframe
########################################################################################################################################################
# Adjust Trade Position
########################################################################################################################################################
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, min_stake: float, max_stake: float, **kwargs):
        if current_profit > self.initial_safety_order_trigger:
            return None
            
        # credits to reinuvader for not blindly executing safety orders
        # Obtain pair dataframe.
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # Only buy when it seems it's climbing back up
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        if last_candle['close'] < previous_candle['close']:
            return None
            
        count_of_buys = 0
        for order in trade.orders:
            if order.ft_is_open or order.ft_order_side != 'enter_long':
                continue
            if order.status == "closed":
                count_of_buys += 1
                
        if 1 <= count_of_buys <= self.max_safety_orders:
        
            safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (math.pow(self.safety_order_step_scale,(count_of_buys - 1)) - 1) / (self.safety_order_step_scale - 1))
            
            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale,(count_of_buys - 1))
                    amount = stake_amount / current_rate
                    logger.info(f"Initiating safety order buy #{count_of_buys} for {trade.pair} with stake amount of {stake_amount} which equals {amount}")
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}') 
                    return None
                    
        return None
########################################################################################################################################################
# Sell Trend
########################################################################################################################################################
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize 'exit_long' to 0 for all rows
        dataframe['exit_long'] = 0
        dataframe['exit_tag'] = 'no_exit'  # Default tag
        
        # Define primary condition based on volume being greater than 0
        primary_condition = dataframe['volume'] > 0
        
        # Define exit conditions
        condition_hma50_above = (
            (dataframe['close'] > dataframe['hma_50']) &
            (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_above.value)) &
            (dataframe['rsi'] > 50) &
            (dataframe['rsi_fast'] > dataframe['rsi_slow'])
        )
        condition_hma50_below = (
            (dataframe['close'] < dataframe['hma_50']) &
            (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_below.value)) &
            (dataframe['rsi_fast'] > dataframe['rsi_slow'])
        )
        
        # Combine conditions with the primary condition
        combined_conditions_above = primary_condition & condition_hma50_above
        combined_conditions_below = primary_condition & condition_hma50_below
        
        # Apply the conditions to set 'exit_long' to 1
        dataframe.loc[combined_conditions_above, 'exit_long'] = 1
        dataframe.loc[combined_conditions_below, 'exit_long'] = 1
        
        # Tagging based on which specific condition was met
        dataframe.loc[combined_conditions_above, 'exit_tag'] = 'hma50_above'
        dataframe.loc[combined_conditions_below, 'exit_tag'] = 'hma50_below'
        
        return dataframe
########################################################################################################################################################
# Confirm Trade Exit
########################################################################################################################################################
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
    
        if trade and trade.exit_reason:
            trade.exit_reason = exit_reason + "_" + trade.enter_tag
    
        return True
########################################################################################################################################################
# Custom to Sell unclog
########################################################################################################################################################
    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # 获取交易时长(分钟)
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60
        
        # unclog退出逻辑优化
        if trade_duration > self.sell_params['unclog_wait_time']:
            # 1. RSI过低时不触发unclog
            if last_candle['rsi'] < self.sell_params['unclog_rsi_threshold']:
                return False
                
            # 2. 价格突破MA时不触发unclog
            if current_rate > last_candle['ma_sell']:
                return False
                
            # 3. 当前亏损超过阈值且无反弹迹象
            if (current_profit < -self.sell_params['unclog_threshold'] and
                last_candle['rsi'] < 40 and
                last_candle['close'] < last_candle['ma_sell']):
                return 'unclog_loss_no_recovery'
                
            # 4. 长期横盘震荡无突破
            if (abs(current_profit) < self.sell_params['unclog_profit_ratio'] and
                trade_duration > self.sell_params['unclog_wait_time'] * 2):
                return 'unclog_sideways'
                
            # 5. 连续下跌且RSI超卖
            if (current_profit < -self.sell_params['unclog_threshold'] * 0.5 and
                last_candle['rsi'] < 35 and
                last_candle['close'] < last_candle['ma_sell'] * 0.995):
                return 'unclog_downtrend'
        
        return None
########################################################################################################################################################
# Trade Protections
########################################################################################################################################################
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]

    @property
    def initial_safety_order_trigger(self) -> float:
        return self.buy_params['initial_safety_order_trigger']