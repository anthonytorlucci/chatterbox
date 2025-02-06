# standard lib
import re

# third party

# local

def is_valid_url(url):
    """
    Checks if the given string is a valid URL.

    Args:
        url: The string to be checked.

    Returns:
        True if the string is a valid URL, False otherwise.

    Example:
        >>> url1 = "https://www.example.com"
        >>> url2 = "invalid_url"
        >>> url3 = "http://192.168.1.1"

        >>> print(f"{url1} is valid: {is_valid_url(url1)}")
        >>> print(f"{url2} is valid: {is_valid_url(url2)}")
        >>> print(f"{url3} is valid: {is_valid_url(url3)}")
    """
    url_regex = re.compile(
        r"^(?:http|https|ftp|ftps)://"  # http:// or https:// or ftp:// or ftps://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]*[A-Z0-9])?\.)+[A-Z]{2,}|"  # domain...
        r"localhost|"  # localhost
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # OR ip address
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$", re.IGNORECASE)
    return url_regex.match(url) is not None
