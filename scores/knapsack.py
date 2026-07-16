"""Compatibility wrapper around the repository's dynamic-programming solver."""

from knapsack import knapsack_dp


def knapsack_ortools(values, weights, items, capacity):
    """Keep the historical API without requiring a version-specific OR-Tools."""
    return knapsack_dp(
        [float(value) for value in values],
        [int(weight) for weight in weights],
        int(items),
        int(capacity),
    )
