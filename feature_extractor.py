import re
from urllib.parse import urlparse


def extract_features(url):
    """Convert a URL into numerical features for the ML model."""

    # Add a scheme if the user enters only a domain
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),

        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equal_signs": url.count("="),
        "num_at_symbols": url.count("@"),

        "has_https": int(parsed.scheme == "https"),
        "has_ip": int(bool(
            re.search(r"(\d{1,3}\.){3}\d{1,3}", hostname)
        )),

        "has_suspicious_words": int(bool(
            re.search(
                r"login|verify|account|password|secure|update|"
                r"confirm|bank|payment|security",
                url.lower()
            )
        )),
    }

    return features