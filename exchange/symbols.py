"""ccxt 统一 symbol 解析：BASE/QUOTE:SETTLE（现货无 SETTLE 段）。

散落各处的 `symbol.split("/")[0] / [-1]` 对合约统一代码会拆出 "USDT:USDT"
这样的报价段——拿它去余额字典里查现金恒为 0，实盘下单数量算不出来就被
静默跳过，对账还会把 quote 币当成一笔未知持仓。统一走这里一次解析。
"""


def parse_symbol(symbol: str) -> tuple[str, str, str]:
    """拆成 (base, quote, settle)；缺失段返回空串而非猜测。

    BTC/USDT          → ("BTC", "USDT", "USDT")
    BTC/USDT:USDT     → ("BTC", "USDT", "USDT")   币安 U 本位永续
    ETH/USDT:USDT     → ("ETH", "USDT", "USDT")
    BTC               → ("BTC", "", "")           非统一代码
    """
    market, _, settle = symbol.partition(":")
    parts = [p for p in market.split("/") if p]
    base = parts[0] if parts else symbol
    quote = parts[1] if len(parts) > 1 else ""
    return base, quote, (settle or quote)


def base_currency(symbol: str) -> str:
    return parse_symbol(symbol)[0]


def quote_currency(symbol: str) -> str:
    return parse_symbol(symbol)[1]
