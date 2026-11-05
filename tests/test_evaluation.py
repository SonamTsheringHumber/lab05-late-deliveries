"""Tests for the cost rule and the threshold choice, with answers worked by hand."""

import pytest

from latedeliveries.evaluation import (
    THRESHOLDS,
    choose_threshold,
    cost_by_threshold,
    order_cost,
    policy_costs,
)


@pytest.mark.parametrize(
    ("y_true", "flagged", "expected"),
    [
        # two vouchers (2 x 4) and one missed late order (1 x 15)
        ([1, 0, 0, 1, 0], [True, True, False, False, False], 2 * 4 + 1 * 15),
        # nobody flagged, two late orders missed
        ([1, 1, 0], [False, False, False], 2 * 15),
        # everybody flagged, three vouchers
        ([1, 0, 0], [True, True, True], 3 * 4),
        # a flagged order that arrives on time still costs $4
        ([0], [True], 4),
        # a late order that was flagged costs only the $4
        ([1], [True], 4),
    ],
)
def test_order_cost_on_small_examples(y_true, flagged, expected):
    assert order_cost(y_true, flagged) == expected


def test_threshold_grid_is_0_05_to_0_95_in_steps_of_0_05():
    assert THRESHOLDS[0] == 0.05
    assert THRESHOLDS[-1] == 0.95
    assert len(THRESHOLDS) == 19


def test_threshold_choice_on_a_hand_made_example():
    # Worked on paper: 2 late orders among 6.
    # t=0.55 flags the two late orders only: 2 x 4 = 8 (best).
    # t=0.25 flags three orders: 12. t=0.95 flags none: 2 x 15 = 30.
    y_true = [1, 1, 0, 0, 0, 0]
    proba = [0.9, 0.6, 0.5, 0.2, 0.1, 0.1]
    costs = cost_by_threshold(y_true, proba, [0.25, 0.55, 0.95])
    assert costs == {0.25: 12, 0.55: 8, 0.95: 30}
    assert choose_threshold(y_true, proba, [0.25, 0.55, 0.95]) == 0.55
    assert choose_threshold(y_true, proba) == 0.55


def test_flagging_nobody_can_be_the_cheapest_policy():
    # One late order in ten, and the model gives it no more than the others.
    y_true = [0] * 9 + [1]
    proba = [0.1] * 10
    best = choose_threshold(y_true, proba)
    assert best == 0.15  # first threshold above every probability
    assert cost_by_threshold(y_true, proba)[best] == 15  # nobody flagged: 1 x 15
    assert cost_by_threshold(y_true, proba)[0.05] == 40  # everybody: 10 x 4


def test_flagging_everybody_can_be_the_cheapest_policy():
    # Every order is late, so a $4 voucher always beats a $15 loss.
    y_true = [1, 1, 1, 1]
    proba = [0.5, 0.5, 0.5, 0.5]
    best = choose_threshold(y_true, proba)
    assert best == 0.05
    assert cost_by_threshold(y_true, proba)[best] == 4 * 4


def test_policy_costs_of_the_three_policies():
    y_true = [1, 0, 0, 1, 0]
    proba = [0.9, 0.6, 0.2, 0.1, 0.1]
    costs = policy_costs(y_true, proba, threshold=0.5)
    assert costs == {
        "flag_nobody": 2 * 15,
        "flag_everybody": 5 * 4,
        "model": 2 * 4 + 1 * 15,  # flags orders 1 and 2; misses the late order 4
    }
