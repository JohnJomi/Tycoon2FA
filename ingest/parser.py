"""RFC-822 to ParsedEmail normalization.

Per ARCHITECTURE.md section 3: stdlib `email` for structure, BeautifulSoup for
the HTML body, URL extraction from <a href>, <img src>, <form action> and
plaintext regex. Parsing is pure and offline - never fetch during parse.

Not implemented - Phase 0 scaffold.
"""
