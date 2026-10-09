"""Run the extracted cryptocurrency ML trade-filter research pipeline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/top100_4h_730d.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--validate-only', action='store_true', help='Check the bundled data and imports without training models')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('ml_trade_filter', ROOT / 'src/ml_trade_filter.py')
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    engine = pipeline.load_backtest_module()
    engine.DATA_PATH = args.data.resolve()
    if engine.DATA_PATH == (ROOT / 'data/top100_4h_730d.csv').resolve():
        manifest = json.loads((ROOT / 'data/DATA_MANIFEST.json').read_text(encoding='utf-8'))
        digest = hashlib.file_digest(engine.DATA_PATH.open('rb'), 'sha256').hexdigest()
        if digest != manifest['sha256']:
            raise ValueError('Bundled market data does not match DATA_MANIFEST.json')
    if args.validate_only:
        raw = engine.load_market_data(engine.DATA_PATH)
        print(json.dumps({'status':'passed','rows':len(raw),'symbols':raw.symbol.nunique(),
            'start':str(raw.dt.min()),'end':str(raw.dt.max()),'bar_hours':engine.BAR_HOURS,
            'models':list(pipeline.model_factories())}, indent=2))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    engine.OUTPUT_DIR = output
    pipeline.ROOT = output
    pipeline.load_backtest_module = lambda: engine
    pipeline.main()


if __name__ == '__main__':
    main()
