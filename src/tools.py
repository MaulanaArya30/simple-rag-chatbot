"""
A deliberately small "tool integration" example.

In a real job, this is where you'd call an internal API (e.g. a fee-rules
service or a ledger system). Here it's a mock lookup so the project stays
runnable without any backend dependency, while still demonstrating the
pattern: the agent decides *when* to call a tool vs. when to retrieve vs.
when to answer directly.
"""

_FEE_TABLE = {
    "qris": "The QRIS merchant fee (MDR) is 0.7% (0.3% for micro-merchants).",
    "checkout": "The in-app checkout fee (MDR) is 1.5%.",
    "cashout_small": "Cash-out up to IDR 100,000 has a flat fee of IDR 2,500.",
    "cashout_medium": (
        "Cash-out between IDR 100,001 and 1,000,000 has a flat fee of IDR 4,000."
    ),
    "cashout_large": (
        "Cash-out above IDR 1,000,000 costs 0.5% of the amount, capped at IDR 25,000."
    ),
}


def lookup_transaction_fee(fee_key: str) -> str:
    """Look up a transaction fee by key.

    Valid keys: qris, checkout, cashout_small, cashout_medium, cashout_large
    """
    return _FEE_TABLE.get(
        fee_key.lower().strip(),
        f"No fee entry found for '{fee_key}'. Valid keys: {list(_FEE_TABLE)}",
    )
