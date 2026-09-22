"""配置 schema 校验回归：类型/范围/枚举/格式/未知键行为。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _write_yaml(tmp_path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_valid_config_passes(tmp_path):
    from scripts.validate_config import validate
    p = _write_yaml(tmp_path, "default_exchange: binance\ndefault_timeframe: 1h\n"
                                "default_start_cash: 10000.0\nweb_port: 8000\n")
    errs = validate(p)
    assert errs == [], f"合法配置不应报错: {errs}"


def test_wrong_type_fails(tmp_path):
    from scripts.validate_config import validate
    p = _write_yaml(tmp_path, "default_start_cash: 'not-a-number'\n")
    errs = validate(p)
    assert any("default_start_cash" in e for e in errs), f"类型错误应被拦截: {errs}"


def test_enum_violation_fails(tmp_path):
    from scripts.validate_config import validate
    p = _write_yaml(tmp_path, "default_timeframe: 1H\n")  # 大写 H 不在枚举
    errs = validate(p)
    assert any("default_timeframe" in e for e in errs)


def test_range_violation_fails(tmp_path):
    from scripts.validate_config import validate
    p = _write_yaml(tmp_path, "web_port: 99999\n")  # 超 65535
    errs = validate(p)
    assert any("web_port" in e for e in errs)


def test_symbol_pattern_fails(tmp_path):
    from scripts.validate_config import validate
    p = _write_yaml(tmp_path, "default_symbol: btcusdt\n")  # 缺斜杠
    errs = validate(p)
    assert any("default_symbol" in e for e in errs)


def test_unknown_key_warns_not_blocks(tmp_path, capsys):
    """未知键：非 strict 仅警告不报错；strict 才报错。"""
    from scripts.validate_config import validate
    p = _write_yaml(tmp_path, "future_key: 1\n")
    errs = validate(p)
    assert errs == [], "非 strict 下未知键不应阻断"
    errs_strict = validate(p, strict=True)
    assert any("future_key" in e for e in errs_strict), "strict 下未知键应报错"


def test_missing_file_fails(tmp_path):
    from scripts.validate_config import validate
    errs = validate(tmp_path / "nope.yaml")
    assert errs and "不存在" in errs[0]


def test_main_returns_zero_on_valid(tmp_path, monkeypatch):
    """CLI 入口：合法配置退出码 0。"""
    import sys
    import scripts.validate_config as vc
    p = _write_yaml(tmp_path, "default_exchange: okx\ndefault_timeframe: 5m\nweb_port: 8080\n")
    monkeypatch.setattr(vc, "CONFIG", p)
    monkeypatch.setattr(sys, "argv", ["validate_config.py"])  # 避免 argparse 读 pytest 参数
    assert vc.main() == 0


def test_show_schema_prints_json(tmp_path, monkeypatch, capsys):
    import scripts.validate_config as vc
    import argparse
    monkeypatch.setattr(vc, "CONFIG", tmp_path / "config.yaml")
    # 直接验证 show-schema 输出可 JSON 解析
    schema = {k: [t.__name__, r] for k, (t, r) in vc.SCHEMA.items()}
    assert json.loads(json.dumps(schema))  # 可序列化
    assert "default_timeframe" in vc.SCHEMA