import sys
sys.path.insert(0, '/datadrive/difan/verl-llm-tandem/verl')


_STYLE_MATH = 'math'
_STYLE_GSM8K = 'gsm8k'

ANSWER_MARKERS = {
    _STYLE_GSM8K: '####',
    _STYLE_MATH: '\\boxed{',
}


def get_answer_marker(style):
    return ANSWER_MARKERS[style]


def make_score_input(answer_text, style):
    if style == _STYLE_GSM8K:
        return f'#### {answer_text}'
    return answer_text


def detect_style(parquet_path):
    import pandas as pd
    df = pd.read_parquet(parquet_path, columns=['data_source'])
    source = str(df['data_source'].iloc[0]).lower()
    if 'gsm8k' in source:
        return _STYLE_GSM8K
    return _STYLE_MATH


def load_grader(style):
    if style == _STYLE_MATH:
        from verl.utils.reward_score.math_dataset import compute_score as _cs
        def _score(response, gt):
            result = _cs(solution_str=response, ground_truth=str(gt))
            return result['score'] if isinstance(result, dict) else float(result)
        return _score
    if style == _STYLE_GSM8K:
        from verl.utils.reward_score.gsm8k import compute_score as _cs
        def _score(response, gt):
            return _cs(response, str(gt), method='strict', format_score=0.1, score=1.0)
        return _score
    raise ValueError(f'Unknown eval style: {style!r}. Use "math" or "gsm8k".')


def resolve_grader(style, parquet_path=None):
    if style == 'auto':
        if parquet_path is None:
            raise ValueError('parquet_path required for style="auto"')
        style = detect_style(parquet_path)
    return load_grader(style), style
