# Hawaii Tax Calculator

Side-by-side comparison of three Hawaii income tax scenarios:

| Scenario | Brackets | CDCC | Refundable? |
|----------|----------|------|-------------|
| Act 46 (2025) | Current law | $3k/$6k caps, 20–25% | No |
| HB 2306 HD1 | Top 3 rates +1pp | $10k/$20k caps, 50%→5% | Yes |
| SB 3125 SD1 | Expanded low/mid brackets | $10k/$20k caps, formula* | Yes |

\* SB 3125 CDCC applicable percentage has blanks in the enacted bill. Calculator assumes 35% at $43k AGI, −1pp per $3k, floor 15%.

## Setup

```bash
pip3 install streamlit plotly
```

Or install all dependencies:

```bash
pip3 install -r requirements.txt
```

## Run

```bash
streamlit run scripts/calculator/tax_calculator_app.py
```

Opens at `http://localhost:8501`.

## Tests

```bash
python3 -m pytest tests/test_tax_calculator_scenarios.py -v
```
