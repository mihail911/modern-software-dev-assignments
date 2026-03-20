import logging, sys

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

import httpx
from fastmcp import FastMCP

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Foggy",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers",
    81: "Moderate showers",
    82: "Violent showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

mcp = FastMCP("weather")


def _fmt_temp(value: float | int | None) -> str:
    if value is None:
        return "?"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "?"
    if abs(f - round(f)) < 1e-6:
        return str(int(round(f)))
    return f"{f:.1f}".rstrip("0").rstrip(".")


async def _geocode(client: httpx.AsyncClient, city: str) -> tuple[float, float, str]:
    try:
        response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError:
        raise
    except (ValueError, TypeError):
        raise httpx.ConnectError("Invalid geocoding response") from None

    results = data.get("results", [])
    if not results:
        raise ValueError(f"City not found: {city}")

    first = results[0]
    lat = first.get("latitude")
    lon = first.get("longitude")
    name = first.get("name")
    if lat is None or lon is None:
        raise ValueError(f"City not found: {city}")
    resolved_name = name if isinstance(name, str) and name.strip() else city
    return (float(lat), float(lon), resolved_name)


@mcp.tool
async def get_current_weather(city: str) -> str:
    """
    Returns the current weather for a given city as a plain English string.

    Args:
        city: Name of the city to look up. Must be a non-empty string.
              Example: "London", "Tokyo", "New York".

    Returns:
        "Current weather in {city}: {temp}°C, {description}."
        Example: "Current weather in London: 10.8°C, Mainly clear."

    Errors returned as strings (server never raises unhandled exceptions):
        "City name cannot be empty. Provide a valid city name."
            — city argument is an empty string
        "City not found: {city}. Check spelling or try a nearby major city."
            — geocoding API returned no results for the given name
        "Weather service unavailable. Please try again shortly."
            — Open-Meteo API is unreachable (timeout or connection error)
        "Could not parse weather data for {city}."
            — API responded but expected fields are missing
    """
    logger.info("get_current_weather called: city=%r", city)
    stripped = city.strip()
    if not stripped:
        logger.error("get_current_weather error: empty city name")
        return "City name cannot be empty. Provide a valid city name."

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                lat, lon, resolved = await _geocode(client, stripped)
            except ValueError as e:
                if str(e).startswith("City not found:"):
                    logger.error("get_current_weather error: City not found: %s", stripped)
                    return (
                        f"City not found: {stripped}. Check spelling or try a nearby major city."
                    )
                raise

            try:
                response = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "timezone": "auto",
                        "current_weather": "true",
                    },
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError:
                raise
            except (ValueError, TypeError):
                raise httpx.ConnectError("Invalid forecast response") from None

            current = data.get("current_weather")
            if not isinstance(current, dict):
                logger.error("get_current_weather error: missing current_weather for %s", stripped)
                return f"Could not parse weather data for {stripped}."

            temp = current.get("temperature")
            code = current.get("weathercode")
            if temp is None or code is None:
                logger.error("get_current_weather error: missing fields for %s", stripped)
                return f"Could not parse weather data for {stripped}."

            try:
                code_int = int(code)
            except (TypeError, ValueError):
                logger.error("get_current_weather error: bad weathercode for %s", stripped)
                return f"Could not parse weather data for {stripped}."

            description = WEATHER_CODES.get(code_int, "Unknown conditions")
            try:
                temp_f = float(temp)
            except (TypeError, ValueError):
                logger.error("get_current_weather error: bad temperature for %s", stripped)
                return f"Could not parse weather data for {stripped}."

            result = (
                f"Current weather in {resolved}: {_fmt_temp(temp_f)}°C, {description}."
            )
            logger.info("get_current_weather returned: %r", result)
            return result

    except httpx.HTTPError as e:
        logger.error("get_current_weather error: %s", e)
        return "Weather service unavailable. Please try again shortly."
    except Exception as e:
        logger.error("get_current_weather error: %s", e)
        return f"Could not parse weather data for {stripped}."


@mcp.tool
async def get_forecast(city: str, days: int = 3) -> str:
    """
    Returns a multi-day weather forecast for a given city as a plain English string.

    Args:
        city: Name of the city to look up. Must be a non-empty string.
              Example: "Tokyo", "Berlin", "Sydney".
        days: How many days of forecast to return. Integer, 1 to 7 inclusive. Default: 3.
              days=1 returns today only. days=7 returns a full week.
              Maximum is 7 — Open-Meteo free tier limit. Values outside 1-7 return an error.

    Returns:
        One line per day: "{date}: High {max}°C / Low {min}°C, {description}."
        Example (3-day):
            "2026-03-20: High 14°C / Low 8°C, Partly cloudy.
             2026-03-21: High 16°C / Low 9°C, Clear sky.
             2026-03-22: High 12°C / Low 7°C, Rain."

    Errors returned as strings (server never raises unhandled exceptions):
        "days must be between 1 and 7. You requested {days}."
            — days < 1 or days > 7. Checked before any API call is made.
        "City name cannot be empty. Provide a valid city name."
            — city argument is an empty string
        "City not found: {city}. Check spelling or try a nearby major city."
            — geocoding API returned no results for the given name
        "Weather service unavailable. Please try again shortly."
            — Open-Meteo API is unreachable (timeout or connection error)
        "Could not parse weather data for {city}."
            — API responded but expected fields are missing
    """
    logger.info("get_forecast called: city=%r, days=%s", city, days)
    if days < 1 or days > 7:
        logger.error("get_forecast error: days out of range — requested %s", days)
        return f"days must be between 1 and 7. You requested {days}."

    stripped = city.strip()
    if not stripped:
        logger.error("get_forecast error: empty city name")
        return "City name cannot be empty. Provide a valid city name."

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                lat, lon, resolved = await _geocode(client, stripped)
            except ValueError as e:
                if str(e).startswith("City not found:"):
                    logger.error("get_forecast error: City not found: %s", stripped)
                    return (
                        f"City not found: {stripped}. Check spelling or try a nearby major city."
                    )
                raise

            try:
                response = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "timezone": "auto",
                        "daily": "temperature_2m_max,temperature_2m_min,weathercode",
                        "forecast_days": days,
                    },
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError:
                raise
            except (ValueError, TypeError):
                raise httpx.ConnectError("Invalid forecast response") from None

            daily = data.get("daily")
            if not isinstance(daily, dict):
                logger.error("get_forecast error: missing daily for %s", stripped)
                return f"Could not parse weather data for {stripped}."

            times = daily.get("time", [])
            maxs = daily.get("temperature_2m_max", [])
            mins = daily.get("temperature_2m_min", [])
            codes = daily.get("weathercode", [])
            if not isinstance(times, list) or not times:
                logger.error("get_forecast error: bad time series for %s", stripped)
                return f"Could not parse weather data for {stripped}."

            lines: list[str] = []
            for i, day in enumerate(times):
                hi = maxs[i] if i < len(maxs) else None
                lo = mins[i] if i < len(mins) else None
                wc = codes[i] if i < len(codes) else None
                if hi is None or lo is None or wc is None:
                    logger.error("get_forecast error: incomplete row %s for %s", i, stripped)
                    return f"Could not parse weather data for {stripped}."
                try:
                    wc_int = int(wc)
                except (TypeError, ValueError):
                    logger.error("get_forecast error: bad weathercode row %s", i)
                    return f"Could not parse weather data for {stripped}."
                try:
                    float(hi)
                    float(lo)
                except (TypeError, ValueError):
                    logger.error("get_forecast error: bad temps row %s", i)
                    return f"Could not parse weather data for {stripped}."
                desc = WEATHER_CODES.get(wc_int, "Unknown conditions")
                day_str = str(day) if day is not None else ""
                lines.append(
                    f"{day_str}: High {_fmt_temp(hi)}°C / Low {_fmt_temp(lo)}°C, {desc}."
                )

            result = "\n".join(lines)
            logger.info("get_forecast returned: %s-day forecast for %s", days, resolved)
            return result

    except httpx.HTTPError as e:
        logger.error("get_forecast error: %s", e)
        return "Weather service unavailable. Please try again shortly."
    except Exception as e:
        logger.error("get_forecast error: %s", e)
        return f"Could not parse weather data for {stripped}."


if __name__ == "__main__":
    mcp.run()
