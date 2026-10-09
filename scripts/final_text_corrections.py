from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'manuscript'
replacements = {
    'abstract_conclusion.txt': [
        ('Across nine physically distinct deployments', 'Across nine distinct simulated deployment cases'),
        ('Voltage benefits depend strongly on source voltage and vanish in two unstressed benchmark networks; a narrower geographic cohort also exhibits increased export peaks.', 'Supplementary comparisons against frozen MSE show strong source-voltage dependence and no narrow-band excursions in either arm on two benchmark networks. A narrower geographic cohort also exhibits increased export peaks.'),
        ('Source voltage changes the measured voltage benefit, unstressed networks have no screening violation to remove, and export peaks can increase.', 'Supplementary comparisons against frozen MSE show that source voltage changes the measured voltage benefit and that two benchmark networks have no narrow-band excursions in either arm. Export peaks can also increase in the narrower geographic cohort.'),
        ('The accompanying research package retains', "The study's computational record retains"),
    ],
    'methods_draft.txt': [
        ('an untouched household test set', 'a household-held-out test group'),
        ('the untouched group', 'the household-held-out group'),
        ('The untouched 55-household panel', 'The household-held-out 55-household panel'),
        ('the untouched 55-household panel', 'the household-held-out 55-household panel'),
        ('The principal physical deployments', 'The principal simulated deployments'),
    ],
    'results_draft.txt': [
        ('nine physically distinct primary deployments', 'nine physically distinct simulated primary deployments'),
        ('the untouched-household panel', 'the household-held-out panel'),
    ],
    'introduction_related_work.txt': [
        ('Across nine primary deployments', 'Across nine simulated primary deployments'),
        ('The joint improvement remains conditional: alternative network settings change voltage exposure, and the scenario baseline can rank differently across cost, import peaks and voltage.', 'The joint improvement remains conditional. Supplementary comparisons against frozen MSE show that alternative network settings change voltage exposure, and the scenario baseline can rank differently across cost, import peaks and voltage.'),
    ],
}
for name, edits in replacements.items():
    p = root / name
    text = p.read_text(encoding='utf-8')
    for old, new in edits:
        assert text.count(old) == 1, (name, old, text.count(old))
        text = text.replace(old, new)
    p.write_text(text, encoding='utf-8')
    print(name, len(edits))
