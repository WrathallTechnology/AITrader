"""
Yahoo Finance options data provider.

Scrapes options chain data from Yahoo Finance and converts to internal models.
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional, Protocol

import requests
from bs4 import BeautifulSoup

from .models import OptionType, OptionContract, OptionChain

logger = logging.getLogger(__name__)


class OptionsDataProvider(Protocol):
    """Protocol for options data providers."""

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
        """Get option chain for an underlying symbol."""
        ...

    def get_underlying_price(self, symbol: str) -> float:
        """Get current price of underlying."""
        ...


@dataclass
class CachedChain:
    """Cached option chain with timestamp."""
    chain: OptionChain
    cached_at: datetime


class YahooOptionsDataProvider:
    """
    Options data provider that scrapes Yahoo Finance.

    Features:
    - Fetches option chains from Yahoo Finance
    - Caches results with configurable TTL (default 5 minutes)
    - Rate limiting to avoid blocks
    - Retry logic with exponential backoff
    """

    BASE_URL = "https://finance.yahoo.com/quote/{symbol}/options"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    def __init__(
        self,
        cache_ttl_minutes: int = 5,
        request_delay_seconds: float = 1.0,
        max_retries: int = 3,
    ):
        """
        Initialize Yahoo options provider.

        Args:
            cache_ttl_minutes: Cache time-to-live in minutes
            request_delay_seconds: Delay between requests to avoid rate limiting
            max_retries: Maximum retry attempts on failure
        """
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.request_delay = request_delay_seconds
        self.max_retries = max_retries

        # Cache: symbol -> CachedChain
        self._chain_cache: dict[str, CachedChain] = {}
        self._price_cache: dict[str, tuple[float, datetime]] = {}

        # Rate limiting
        self._last_request_time: datetime = datetime.min

        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)

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
        Get option chain from Yahoo Finance.

        Args:
            underlying: Stock symbol (e.g., "MU", "RKLB")
            expiration_date: Specific expiration date
            expiration_date_gte: Minimum expiration date
            expiration_date_lte: Maximum expiration date
            strike_price_gte: Minimum strike price
            strike_price_lte: Maximum strike price
            option_type: Filter by call or put

        Returns:
            OptionChain with contracts matching criteria
        """
        underlying = underlying.upper()

        # Check cache first
        cache_key = underlying
        if cache_key in self._chain_cache:
            cached = self._chain_cache[cache_key]
            if datetime.now() - cached.cached_at < self.cache_ttl:
                logger.debug(f"Using cached chain for {underlying}")
                return self._filter_chain(
                    cached.chain,
                    expiration_date,
                    expiration_date_gte,
                    expiration_date_lte,
                    strike_price_gte,
                    strike_price_lte,
                    option_type,
                )

        # Fetch fresh data
        try:
            chain = self._fetch_full_chain(underlying)

            # Update cache
            self._chain_cache[cache_key] = CachedChain(
                chain=chain,
                cached_at=datetime.now(),
            )

            # Apply filters
            return self._filter_chain(
                chain,
                expiration_date,
                expiration_date_gte,
                expiration_date_lte,
                strike_price_gte,
                strike_price_lte,
                option_type,
            )

        except Exception as e:
            logger.error(f"Failed to fetch option chain for {underlying}: {e}")
            return OptionChain(
                underlying=underlying,
                underlying_price=0,
                expirations=[],
                contracts=[],
            )

    def get_underlying_price(self, symbol: str) -> float:
        """Get current price of underlying from Yahoo Finance."""
        symbol = symbol.upper()

        # Check cache (1 minute TTL for prices)
        if symbol in self._price_cache:
            price, cached_at = self._price_cache[symbol]
            if datetime.now() - cached_at < timedelta(minutes=1):
                return price

        try:
            url = f"https://finance.yahoo.com/quote/{symbol}"
            html = self._fetch_with_retry(url)
            soup = BeautifulSoup(html, "lxml")

            # Find price in the page - look for the main price display
            price = self._extract_price_from_html(soup)
            if price > 0:
                self._price_cache[symbol] = (price, datetime.now())
                logger.info(f"Got underlying price for {symbol}: ${price:.2f}")
                return price

        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")

        return 0.0

    def _extract_price_from_html(self, soup: BeautifulSoup, symbol: str) -> float:
        """Extract stock price from Yahoo Finance HTML."""
        symbol = symbol.upper()

        # Method 1: Find fin-streamer with matching symbol
        streamers = soup.find_all("fin-streamer", {"data-field": "regularMarketPrice"})
        for streamer in streamers:
            data_symbol = streamer.get("data-symbol", "").upper()
            if data_symbol == symbol:
                try:
                    value = streamer.get("data-value")
                    if value:
                        return float(value)
                except (ValueError, TypeError):
                    continue

        # Method 2: Look for span with qsp-price (usually the main quote price)
        price_span = soup.find("span", {"data-testid": "qsp-price"})
        if price_span:
            try:
                text = price_span.get_text(strip=True).replace(",", "")
                if text:
                    return float(text)
            except (ValueError, TypeError):
                pass

        # Method 3: Look for the quote price in the header section
        # Yahoo often has the price in a specific container
        quote_header = soup.find("section", {"data-testid": "quote-header"})
        if quote_header:
            streamer = quote_header.find("fin-streamer", {"data-field": "regularMarketPrice"})
            if streamer:
                try:
                    value = streamer.get("data-value")
                    if value:
                        return float(value)
                except (ValueError, TypeError):
                    pass

        return 0.0

    def _fetch_full_chain(self, symbol: str) -> OptionChain:
        """Fetch complete option chain for all expirations."""
        logger.info(f"Fetching option chain for {symbol} from Yahoo Finance")

        # Fetch the options page
        base_url = self.BASE_URL.format(symbol=symbol)
        html = self._fetch_with_retry(base_url)
        soup = BeautifulSoup(html, "lxml")

        # Get underlying price
        underlying_price = self._extract_price_from_html(soup, symbol)
        logger.info(f"Got underlying price for {symbol}: ${underlying_price:.2f}")

        # Parse contracts from the current page (default expiration)
        # The expiration will be extracted from contract names
        all_contracts = self._parse_options_table_v2(html, symbol)

        if not all_contracts:
            logger.warning(f"No contracts found for {symbol}")
            return OptionChain(
                underlying=symbol,
                underlying_price=underlying_price,
                expirations=[],
                contracts=[],
            )

        # Extract unique expirations from contracts
        expirations = sorted(set(c.expiration for c in all_contracts))
        logger.info(f"Found {len(all_contracts)} contracts for {symbol} with {len(expirations)} expirations")

        return OptionChain(
            underlying=symbol,
            underlying_price=underlying_price,
            expirations=expirations,
            contracts=all_contracts,
        )

    def _fetch_with_retry(self, url: str) -> str:
        """Fetch URL with retry logic and rate limiting."""
        # Rate limiting
        elapsed = (datetime.now() - self._last_request_time).total_seconds()
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                self._last_request_time = datetime.now()
                response = self._session.get(url, timeout=15)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                last_error = e
                wait_time = (2 ** attempt)  # Exponential backoff
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}, retrying in {wait_time}s")
                time.sleep(wait_time)

        raise last_error or Exception("Unknown error during fetch")

    def _parse_expiration_dates(self, html: str) -> list[date]:
        """Parse available expiration dates from the options page."""
        soup = BeautifulSoup(html, "lxml")
        expirations = []

        # Method 1: Look for select dropdown with expiration dates
        for select in soup.find_all("select"):
            for option in select.find_all("option"):
                try:
                    value = option.get("value", "")
                    if value and value.isdigit():
                        epoch = int(value)
                        if 1500000000 < epoch < 2500000000:
                            exp_date = datetime.fromtimestamp(epoch).date()
                            if exp_date > date.today():
                                expirations.append(exp_date)
                except (ValueError, TypeError, OSError):
                    continue

        # Method 2: Look for data attributes with timestamps
        if not expirations:
            for elem in soup.find_all(attrs={"data-value": True}):
                try:
                    value = elem.get("data-value", "")
                    if value and str(value).isdigit():
                        epoch = int(value)
                        if 1500000000 < epoch < 2500000000:
                            exp_date = datetime.fromtimestamp(epoch).date()
                            if exp_date > date.today():
                                expirations.append(exp_date)
                except (ValueError, TypeError, OSError):
                    continue

        # Method 3: Parse from URL patterns in the page
        if not expirations:
            pattern = r'date=(\d{10})'
            for match in re.findall(pattern, html):
                try:
                    epoch = int(match)
                    exp_date = datetime.fromtimestamp(epoch).date()
                    if exp_date > date.today():
                        expirations.append(exp_date)
                except (ValueError, OSError):
                    continue

        return sorted(set(expirations))

    def _parse_options_table(
        self,
        html: str,
        symbol: str,
        expiration: date,
    ) -> list[OptionContract]:
        """Parse options table from Yahoo Finance HTML."""
        soup = BeautifulSoup(html, "lxml")
        contracts = []

        # Find all tables - typically calls first, then puts
        tables = soup.find_all("table")

        for i, table in enumerate(tables[:2]):  # Only process first 2 tables
            # Determine if calls or puts based on position
            option_type = OptionType.CALL if i == 0 else OptionType.PUT

            # Find table body
            tbody = table.find("tbody")
            if not tbody:
                continue

            for row in tbody.find_all("tr"):
                try:
                    contract = self._parse_table_row(row, symbol, expiration, option_type)
                    if contract:
                        contracts.append(contract)
                except Exception as e:
                    logger.debug(f"Failed to parse row: {e}")

        return contracts

    def _parse_options_table_v2(self, html: str, symbol: str) -> list[OptionContract]:
        """Parse options table, extracting expiration from contract names."""
        soup = BeautifulSoup(html, "lxml")
        contracts = []

        # Find all tables - typically calls first, then puts
        tables = soup.find_all("table")
        logger.debug(f"Found {len(tables)} tables for {symbol}")

        for i, table in enumerate(tables[:2]):  # Only process first 2 tables
            # Determine if calls or puts based on position
            option_type = OptionType.CALL if i == 0 else OptionType.PUT

            # Find table body
            tbody = table.find("tbody")
            if not tbody:
                # Try finding rows directly in table
                rows = table.find_all("tr")[1:]  # Skip header
            else:
                rows = tbody.find_all("tr")

            for row in rows:
                try:
                    contract = self._parse_table_row_v2(row, symbol, option_type)
                    if contract:
                        contracts.append(contract)
                except Exception as e:
                    logger.debug(f"Failed to parse row: {e}")

        return contracts

    def _parse_table_row_v2(
        self,
        row,
        symbol: str,
        option_type: OptionType,
    ) -> Optional[OptionContract]:
        """Parse a single row, extracting expiration from contract name."""
        cells = row.find_all("td")
        if len(cells) < 10:
            return None

        try:
            # Column 0: Contract Name (e.g., MU260130C00035000)
            contract_name = cells[0].get_text(strip=True)

            # Extract expiration from contract name
            # Format: SYMBOL + YYMMDD + C/P + STRIKE
            match = re.match(rf'^{symbol}(\d{{6}})[CP](\d{{8}})$', contract_name.upper())
            if not match:
                # Try more flexible match
                match = re.search(rf'{symbol}(\d{{6}})([CP])(\d{{8}})', contract_name.upper())
                if not match:
                    return None

            date_str = match.group(1)  # YYMMDD
            expiration = datetime.strptime(date_str, "%y%m%d").date()

            # Parse other columns
            strike = self._parse_number(cells[2].get_text(strip=True))
            last_price = self._parse_number(cells[3].get_text(strip=True))
            bid = self._parse_number(cells[4].get_text(strip=True))
            ask = self._parse_number(cells[5].get_text(strip=True))
            volume = self._parse_int(cells[8].get_text(strip=True))
            open_interest = self._parse_int(cells[9].get_text(strip=True))

            # Parse IV if available
            iv = None
            if len(cells) > 10:
                iv_text = cells[10].get_text(strip=True)
                iv = self._parse_percent(iv_text)

            if strike <= 0:
                return None

            # Use the contract name as-is for the OCC symbol
            occ_symbol = contract_name.upper()

            return OptionContract(
                symbol=occ_symbol,
                underlying=symbol,
                option_type=option_type,
                strike=strike,
                expiration=expiration,
                bid=bid,
                ask=ask,
                last=last_price,
                volume=volume,
                open_interest=open_interest,
                implied_volatility=iv,
            )

        except Exception as e:
            logger.debug(f"Failed to parse contract row: {e}")
            return None

    def _parse_table_row(
        self,
        row,
        symbol: str,
        expiration: date,
        option_type: OptionType,
    ) -> Optional[OptionContract]:
        """Parse a single row from the options table."""
        cells = row.find_all("td")
        if len(cells) < 10:
            return None

        try:
            # Yahoo table columns (typical layout):
            # 0: Contract Name, 1: Last Trade Date, 2: Strike, 3: Last Price,
            # 4: Bid, 5: Ask, 6: Change, 7: % Change, 8: Volume, 9: Open Interest, 10: IV

            contract_name = cells[0].get_text(strip=True)
            strike = self._parse_number(cells[2].get_text(strip=True))
            last_price = self._parse_number(cells[3].get_text(strip=True))
            bid = self._parse_number(cells[4].get_text(strip=True))
            ask = self._parse_number(cells[5].get_text(strip=True))
            volume = self._parse_int(cells[8].get_text(strip=True))
            open_interest = self._parse_int(cells[9].get_text(strip=True))

            # Parse IV (remove % sign) - may not always be present
            iv = None
            if len(cells) > 10:
                iv_text = cells[10].get_text(strip=True)
                iv = self._parse_percent(iv_text)

            # Skip invalid contracts
            if strike <= 0:
                return None

            # Build OCC symbol
            occ_symbol = self._build_occ_symbol(symbol, expiration, option_type, strike)

            return OptionContract(
                symbol=occ_symbol,
                underlying=symbol,
                option_type=option_type,
                strike=strike,
                expiration=expiration,
                bid=bid,
                ask=ask,
                last=last_price,
                volume=volume,
                open_interest=open_interest,
                implied_volatility=iv,
            )

        except Exception as e:
            logger.debug(f"Failed to parse contract row: {e}")
            return None

    def _build_occ_symbol(
        self,
        underlying: str,
        expiration: date,
        option_type: OptionType,
        strike: float,
    ) -> str:
        """Build OCC-format option symbol."""
        # Format: SYMBOL + YYMMDD + C/P + Strike*1000 (8 digits, zero-padded)
        date_str = expiration.strftime("%y%m%d")
        type_char = "C" if option_type == OptionType.CALL else "P"
        strike_int = int(strike * 1000)
        return f"{underlying}{date_str}{type_char}{strike_int:08d}"

    def _parse_number(self, text: str) -> float:
        """Parse number from text, handling commas and dashes."""
        text = text.replace(",", "").replace("-", "0").strip()
        try:
            return float(text) if text else 0.0
        except ValueError:
            return 0.0

    def _parse_int(self, text: str) -> int:
        """Parse integer from text."""
        return int(self._parse_number(text))

    def _parse_percent(self, text: str) -> Optional[float]:
        """Parse percentage to decimal (e.g., '45.5%' -> 0.455)."""
        text = text.replace("%", "").replace(",", "").strip()
        try:
            return float(text) / 100 if text else None
        except ValueError:
            return None

    def _filter_chain(
        self,
        chain: OptionChain,
        expiration_date: Optional[date],
        expiration_date_gte: Optional[date],
        expiration_date_lte: Optional[date],
        strike_price_gte: Optional[float],
        strike_price_lte: Optional[float],
        option_type: Optional[OptionType],
    ) -> OptionChain:
        """Filter chain contracts based on criteria."""
        contracts = chain.contracts

        if expiration_date:
            contracts = [c for c in contracts if c.expiration == expiration_date]
        if expiration_date_gte:
            contracts = [c for c in contracts if c.expiration >= expiration_date_gte]
        if expiration_date_lte:
            contracts = [c for c in contracts if c.expiration <= expiration_date_lte]
        if strike_price_gte:
            contracts = [c for c in contracts if c.strike >= strike_price_gte]
        if strike_price_lte:
            contracts = [c for c in contracts if c.strike <= strike_price_lte]
        if option_type:
            contracts = [c for c in contracts if c.option_type == option_type]

        # Filter expirations too
        expirations = sorted(set(c.expiration for c in contracts))

        return OptionChain(
            underlying=chain.underlying,
            underlying_price=chain.underlying_price,
            expirations=expirations,
            contracts=contracts,
            timestamp=chain.timestamp,
        )

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data."""
        if symbol:
            symbol = symbol.upper()
            self._chain_cache.pop(symbol, None)
            self._price_cache.pop(symbol, None)
        else:
            self._chain_cache.clear()
            self._price_cache.clear()
