import pytest

from kappaml_core.cli import fib, main

__author__ = "Alex Imbrea"
__copyright__ = "Alex Imbrea"
__license__ = "Apache-2.0"


def test_fib():
    """API Tests"""
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)


def test_main_fib(capsys):
    """CLI Test Fib Command"""
    main(["fib", "7"])
    captured = capsys.readouterr()
    assert "The 7-th Fibonacci number is 13" in captured.out


def test_main_demo(capsys):
    """CLI Test Demo Command"""
    main(["demo", "baseline"])
    captured = capsys.readouterr()
    assert "Baseline model" in captured.out
    main(["demo", "greedy"])
    captured = capsys.readouterr()
    assert "Greedy model selection" in captured.out
    main(["demo", "epsilon_greedy"])
    captured = capsys.readouterr()
    assert "Epsilon-Greedy model selection" in captured.out
