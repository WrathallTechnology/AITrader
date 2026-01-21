"""
Options client for Alpaca API.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    OrderType,
    TimeInForce,
    AssetClass,
)
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionLatestQuoteRequest

from .models import (
    OptionType,
    OptionContract,
    OptionChain,
    OptionGreeks,
    OptionPosition,
    OptionOrder,
    OrderAction,
)

logger = logging.getLogger(__name__)


class OptionsClient:
    """
    Client for trading options via Alpaca.

    Note: Alpaca options trading requires specific account approval.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
    ):
        """
        Initialize options client.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading (default True)
        """
        self.paper = paper

        # Trading client for orders
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

        # Data client for quotes and chains
        self.data_client = OptionHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

    def get_option_chain(
        self,
        underlying: str,
        expiration_date: Optional[date] = None,
        expiration_date_gte: Optional[date] = None,
        expiration_date_lte: Optional[date] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
        option_type: Optional[OptionType] = None,
    ) -> OptionChain:
        """
        Get option chain for an underlying symbol.

        Args:
            underlying: Underlying symbol (e.g., "AAPL")
            expiration_date: Specific expiration date
            expiration_date_gte: Minimum expiration date
            expiration_date_lte: Maximum expiration date
            strike_price_gte: Minimum strike price
            strike_price_lte: Maximum strike price
            option_type: Filter by call or put

        Returns:
            OptionChain with available contracts
        """
        try:
            # Build request
            request_params = {
                "underlying_symbol": underlying,
            }

            if expiration_date:
                request_params["expiration_date"] = expiration_date
            if expiration_date_gte:
                request_params["expiration_date_gte"] = expiration_date_gte
            if expiration_date_lte:
                request_params["expiration_date_lte"] = expiration_date_lte
            if strike_price_gte:
                request_params["strike_price_gte"] = strike_price_gte
            if strike_price_lte:
                request_params["strike_price_lte"] = strike_price_lte
            if option_type:
                request_params["type"] = option_type.value

            request = GetOptionContractsRequest(**request_params)
            logger.info(f"Requesting option chain for {underlying} with params: {request_params}")
            response = self.trading_client.get_option_contracts(request)
            logger.info(f"API returned {len(response.option_contracts or [])} contracts for {underlying}")

            # Get underlying price
            underlying_price = self._get_underlying_price(underlying)

            # Parse contracts
            contracts = []
            expirations = set()

            for contract_data in response.option_contracts or []:
                contract = self._parse_contract(contract_data)
                if contract:
                    # Validate contract matches requested underlying
                    if contract.underlying.upper() != underlying.upper():
                        logger.warning(f"API returned contract for {contract.underlying}, expected {underlying} - skipping")
                        continue
                    contracts.append(contract)
                    expirations.add(contract.expiration)

            # Get quotes for contracts
            if contracts:
                self._update_quotes(contracts)

            return OptionChain(
                underlying=underlying,
                underlying_price=underlying_price,
                expirations=sorted(expirations),
                contracts=contracts,
            )

        except Exception as e:
            logger.error(f"Failed to get option chain for {underlying}: {e}")
            return OptionChain(
                underlying=underlying,
                underlying_price=0,
                expirations=[],
                contracts=[],
            )

    def _parse_contract(self, contract_data) -> Optional[OptionContract]:
        """Parse Alpaca contract data into our model."""
        try:
            # Ensure open_interest is an int (API may return string or None)
            open_interest = contract_data.open_interest
            if open_interest is None:
                open_interest = 0
            elif isinstance(open_interest, str):
                open_interest = int(open_interest) if open_interest.isdigit() else 0
            else:
                open_interest = int(open_interest)

            return OptionContract(
                symbol=contract_data.symbol,
                underlying=contract_data.underlying_symbol,
                option_type=OptionType(contract_data.type.lower()),
                strike=float(contract_data.strike_price),
                expiration=contract_data.expiration_date,
                open_interest=open_interest,
            )
        except Exception as e:
            logger.debug(f"Failed to parse contract: {e}")
            return None

    def _update_quotes(self, contracts: list[OptionContract]) -> None:
        """Update quotes for a list of contracts."""
        try:
            symbols = [c.symbol for c in contracts]

            # Get latest quotes
            request = OptionLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self.data_client.get_option_latest_quote(request)

            # Update contracts with quote data
            quotes_found = 0
            valid_quotes = 0
            for contract in contracts:
                if contract.symbol in quotes:
                    quotes_found += 1
                    quote = quotes[contract.symbol]
                    bid = float(quote.bid_price or 0)
                    ask = float(quote.ask_price or 0)
                    contract.bid = bid
                    contract.ask = ask
                    # Track valid quotes
                    if bid > 0 and ask > 0 and bid < ask:
                        valid_quotes += 1

            logger.debug(f"_update_quotes: {len(contracts)} contracts, {quotes_found} quotes found, {valid_quotes} valid (bid>0, ask>0, bid<ask)")

        except Exception as e:
            logger.debug(f"Failed to update quotes: {e}")

    def _get_underlying_price(self, symbol: str) -> float:
        """Get current price of underlying."""
        try:
            # Use trading client to get latest trade with IEX feed (free tier)
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestTradeRequest
            from alpaca.data.enums import DataFeed

            stock_client = StockHistoricalDataClient(
                api_key=self.trading_client._api_key,
                secret_key=self.trading_client._secret_key,
            )

            # Use IEX feed for free tier compatibility
            request = StockLatestTradeRequest(
                symbol_or_symbols=symbol,
                feed=DataFeed.IEX,
            )
            trades = stock_client.get_stock_latest_trade(request)

            if symbol in trades:
                price = float(trades[symbol].price)
                logger.debug(f"Got underlying price for {symbol}: ${price:.2f}")
                return price

        except Exception as e:
            logger.warning(f"Failed to get underlying price for {symbol}: {e}")

        return 0.0

    def get_positions(self) -> list[OptionPosition]:
        """Get all open option positions."""
        try:
            positions = self.trading_client.get_all_positions()

            option_positions = []
            for pos in positions:
                # Filter for options (asset_class check)
                if hasattr(pos, 'asset_class') and pos.asset_class == AssetClass.US_OPTION:
                    # Parse the option symbol to get contract details
                    contract = self._symbol_to_contract(pos.symbol)
                    if contract:
                        option_positions.append(OptionPosition(
                            contract=contract,
                            quantity=int(pos.qty),
                            avg_cost=float(pos.avg_entry_price),
                        ))

            return option_positions

        except Exception as e:
            logger.error(f"Failed to get option positions: {e}")
            return []

    def _symbol_to_contract(self, symbol: str) -> Optional[OptionContract]:
        """Parse OCC symbol to OptionContract."""
        try:
            # OCC format: AAPL230120C00150000
            # Underlying: variable length, ends where date starts
            # Date: 6 digits (YYMMDD)
            # Type: C or P
            # Strike: 8 digits (5 integer, 3 decimal, no decimal point)

            # Find where the date starts (first digit after letters)
            i = 0
            while i < len(symbol) and not symbol[i].isdigit():
                i += 1

            underlying = symbol[:i]
            rest = symbol[i:]

            # Parse date (6 chars)
            date_str = rest[:6]
            exp_date = datetime.strptime(date_str, "%y%m%d").date()

            # Parse type
            opt_type_char = rest[6]
            opt_type = OptionType.CALL if opt_type_char == 'C' else OptionType.PUT

            # Parse strike (8 chars, divide by 1000)
            strike_str = rest[7:15]
            strike = int(strike_str) / 1000

            return OptionContract(
                symbol=symbol,
                underlying=underlying,
                option_type=opt_type,
                strike=strike,
                expiration=exp_date,
            )

        except Exception as e:
            logger.debug(f"Failed to parse option symbol {symbol}: {e}")
            return None

    def submit_order(self, order: OptionOrder) -> Optional[str]:
        """
        Submit an option order.

        Args:
            order: OptionOrder to submit

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Determine side
            if order.is_buy:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL

            # Create order request
            if order.order_type == "market":
                request = MarketOrderRequest(
                    symbol=order.contract.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce(order.time_in_force.upper()),
                )
            else:
                if order.limit_price is None:
                    mid = order.contract.mid_price
                    if mid is None:
                        logger.error(f"Cannot submit limit order - no valid mid price for {order.contract.symbol}")
                        return None
                    order.limit_price = mid

                request = LimitOrderRequest(
                    symbol=order.contract.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce(order.time_in_force.upper()),
                    limit_price=order.limit_price,
                )

            # Submit order
            response = self.trading_client.submit_order(request)
            logger.info(f"Submitted option order: {response.id}")
            return str(response.id)

        except Exception as e:
            logger.error(f"Failed to submit option order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_account_info(self) -> dict:
        """Get account information relevant for options trading."""
        try:
            account = self.trading_client.get_account()
            return {
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "options_buying_power": float(getattr(account, 'options_buying_power', account.buying_power)),
                "options_approved_level": getattr(account, 'options_approved_level', 0),
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    def find_contracts(
        self,
        underlying: str,
        option_type: OptionType,
        min_days: int = 7,
        max_days: int = 45,
        delta_target: Optional[float] = None,
        min_open_interest: int = 10,  # Lowered from 100 to 10
        max_spread_pct: float = 0.20,  # Relaxed from 0.10 to 0.20
    ) -> list[OptionContract]:
        """
        Find suitable option contracts based on criteria.

        Args:
            underlying: Underlying symbol
            option_type: CALL or PUT
            min_days: Minimum days to expiration
            max_days: Maximum days to expiration
            delta_target: Target delta (e.g., 0.30 for 30-delta)
            min_open_interest: Minimum open interest
            max_spread_pct: Maximum bid-ask spread percentage

        Returns:
            List of matching contracts sorted by relevance
        """
        # Get chain
        exp_gte = date.today() + timedelta(days=min_days)
        exp_lte = date.today() + timedelta(days=max_days)

        chain = self.get_option_chain(
            underlying=underlying,
            expiration_date_gte=exp_gte,
            expiration_date_lte=exp_lte,
            option_type=option_type,
        )

        logger.debug(f"find_contracts: {underlying} {option_type.value} - got {len(chain.contracts)} contracts")

        # Filter contracts
        suitable = []
        filtered_not_tradeable = 0
        filtered_oi = 0
        filtered_spread = 0

        for contract in chain.contracts:
            # First check: contract must be tradeable (valid bid/ask)
            if not contract.is_tradeable:
                # Allow contracts without quotes only if they have good open interest
                if contract.open_interest < min_open_interest:
                    filtered_not_tradeable += 1
                    continue

            # Check open interest
            if contract.open_interest < min_open_interest:
                filtered_oi += 1
                continue

            # Check spread (only for tradeable contracts)
            spread_pct = contract.spread_pct
            if spread_pct is not None and spread_pct > max_spread_pct:
                filtered_spread += 1
                continue

            suitable.append(contract)

        logger.debug(f"find_contracts: {underlying} - filtered not_tradeable:{filtered_not_tradeable}, OI:{filtered_oi}, spread:{filtered_spread}, remaining:{len(suitable)}")

        # Sort by delta closeness if target specified
        if delta_target and suitable:
            def delta_distance(c):
                if c.greeks:
                    return abs(abs(c.greeks.delta) - delta_target)
                # Estimate delta from moneyness
                moneyness = c.moneyness(chain.underlying_price)
                estimated_delta = 0.5 if moneyness == 1 else (0.7 if moneyness > 1 else 0.3)
                return abs(estimated_delta - delta_target)

            suitable.sort(key=delta_distance)
        else:
            # Sort by liquidity (volume * open_interest)
            suitable.sort(key=lambda c: (c.volume or 0) * (c.open_interest or 0), reverse=True)

        return suitable

    def calculate_max_profit_loss(
        self,
        order: OptionOrder,
        underlying_price: float,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate max profit and max loss for an order.

        Returns:
            Tuple of (max_profit, max_loss)
        """
        contract = order.contract
        price = order.limit_price or contract.mid_price
        if price is None:
            return None, None
        cost = price * order.quantity * contract.multiplier

        if order.action == OrderAction.BUY_TO_OPEN:
            # Long option
            max_loss = cost
            if contract.option_type == OptionType.CALL:
                max_profit = None  # Unlimited for calls
            else:
                max_profit = (contract.strike * contract.multiplier * order.quantity) - cost

        elif order.action == OrderAction.SELL_TO_OPEN:
            # Short option
            max_profit = cost  # Premium received
            if contract.option_type == OptionType.CALL:
                max_loss = None  # Unlimited for naked calls
            else:
                # Cash secured put: max loss = strike - premium
                max_loss = (contract.strike * contract.multiplier * order.quantity) - cost

        else:
            # Closing orders
            return None, None

        return max_profit, max_loss
