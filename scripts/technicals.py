def add_technical_indicators(df, indicators):
    if 'rsi' in indicators:
        df['rsi'] = calculate_rsi(df['Close'])
    if 'macd' in indicators:
        df['macd'] = calculate_macd(df['Close'])
    if 'sma' in indicators:
        df['sma'] = df['Close'].rolling(window=20).mean()
    if 'ema' in indicators:
        df['ema'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df


def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd - signal
