import pytest
from behavior.attention_scorer import AttentionScorer
from behavior.focus_scorer import FocusScorer
from behavior.flag_detector import FlagDetector


def test_attention_perfect():
    scorer = AttentionScorer()
    score = scorer.score({"yaw": 0, "pitch": 0, "roll": 0}, {"ear_left": 0.3, "ear_right": 0.3})
    assert score == 100.0


def test_attention_eyes_closed():
    scorer = AttentionScorer()
    score = scorer.score({"yaw": 0, "pitch": 0, "roll": 0}, {"ear_left": 0.15, "ear_right": 0.15})
    assert score <= 60.0


def test_attention_no_face():
    scorer = AttentionScorer()
    assert scorer.score(None, None) == 0.0


def test_flag_phone():
    fd = FlagDetector()
    flags = fd.detect({"ear_left": 0.3, "ear_right": 0.3}, {"phone": True}, {"landmarks": []})
    assert flags["phone_detected"] is True


def test_flag_left_desk():
    fd = FlagDetector()
    flags = fd.detect(None, {}, None)
    assert flags["left_desk"] is True
