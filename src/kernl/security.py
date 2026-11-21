import keyring

SERVICE = "kernl"


def set_secret(provider: str, secret: str) -> None:
    if not secret:
        raise ValueError("secret cannot be empty")
    keyring.set_password(SERVICE, provider, secret)


def get_secret(provider: str) -> str | None:
    """Return a stored secret for validation, or None when it is absent."""
    return keyring.get_password(SERVICE, provider) or None


def has_secret(provider: str) -> bool:
    return get_secret(provider) is not None
