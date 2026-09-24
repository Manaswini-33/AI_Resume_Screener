from src.robustness.robustness_tests import detect_keyword_stuffing

def test_keyword_stuffing_detection():
    normal_text = "Software Engineer with experience in Python and SQL."
    stuffed_text = "Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python"

    normal_res = detect_keyword_stuffing(normal_text)
    assert normal_res["is_suspicious"] is False

    stuffed_res = detect_keyword_stuffing(stuffed_text)
    assert stuffed_res["is_suspicious"] is True
