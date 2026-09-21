def test_codecs_import():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.codecs import CODEC_SPECS
    assert "amr_nb_4_75" in CODEC_SPECS
    assert "amr_wb_12_65" in CODEC_SPECS
    assert "gsm" in CODEC_SPECS
    assert "opus_8" in CODEC_SPECS
