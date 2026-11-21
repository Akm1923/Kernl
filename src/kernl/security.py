import keyring

SERVICE = "kernl"


def set_secret(provider: str, secret: str) -> None:
    if not secret:
        raise ValueError("secret cannot be empty")
    keyring.set_password(SERVICE, provider, secret)


def has_secret(provider: str) -> bool:
    return bool(keyring.get_password(SERVICE, provider))
