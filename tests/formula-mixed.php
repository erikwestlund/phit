<?php
declare(strict_types=1);
// stats_lmer/stats_glmer formulas versus lme4 (tests/generate-formula-lme4-fixtures.R).
// Tolerances follow lmm-expanded-r (lmer) and logistic-mixed-r (glmer).
function mcheck(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}
$comparisons = 0;
function mcompare(mixed $actual, mixed $expected, string $label, float $abs, float $rel): void {
    global $comparisons;
    if (is_array($expected)) {
        mcheck(is_array($actual) && count($actual) === count($expected), "$label shape");
        foreach ($expected as $i => $value) mcompare($actual[$i] ?? null, $value, "$label/$i", $abs, $rel);
        return;
    }
    mcheck(abs($actual - $expected) <= $abs + $rel * abs($expected), "$label: $actual != $expected");
    $comparisons++;
}
$fixture = json_decode(file_get_contents(__DIR__ . '/fixtures/formula-lme4-r.json'), true, flags: JSON_THROW_ON_ERROR);
$fits = ['lmer' => 0, 'glmer' => 0];
foreach ($fixture['cases'] as $case) {
    $data = $fixture['datasets'][$case['dataset']];
    $name = $case['name'];
    $result = @($case['function'] === 'lmer'
        ? stats_lmer($case['formula'], $data, reml: $case['reml'])
        : stats_glmer($case['formula'], $data, quadrature_points: $case['quadrature_points']));
    $expected = $case['expected'];
    mcheck($result['columns'] === $expected['columns'], "$name: columns");
    mcheck($result['n_observations'] === $expected['n_observations'], "$name: observations");
    mcheck(array_column($result['random'], 'group') === array_column($case['bars'], 'group'), "$name: random groups");
    mcheck(array_map(fn($t) => (array)$t['columns'], $result['random']) === array_map(fn($b) => (array)$b['columns'], $case['bars']), "$name: random columns");
    foreach ($expected as $key => $value) {
        if (in_array($key, ['columns', 'n_observations'], true)) continue;
        $likelihood = in_array($key, ['log_likelihood', 'aic', 'bic'], true);
        if ($case['function'] === 'lmer') {
            [$abs, $rel] = $likelihood ? [5e-6, 1e-9] : [1e-5, 1e-5];
            $actual = $key === 'random_covariances' ? array_column($result['random'], 'covariance') : $result[$key];
        } else {
            [$abs, $rel] = $likelihood ? [1e-6, 1e-8] : (in_array($key, ['std_errors', 'covariance'], true) ? [2e-5, 2e-3] : [2e-5, 2e-4]);
            $actual = $result[$key];
        }
        mcompare($actual, $value, "$name/$key", $abs, $rel);
    }
    mcheck(array_keys($result['named']) === $expected['columns'], "$name: named");
    $fits[$case['function']]++;
}

// Formula fits equal the matrix interfaces they wrap.
$schools = $fixture['datasets']['schools'];
$lmer = stats_lmer('score ~ study_hours + (1 | school)', $schools, reml: false);
$matrix = stats_lmm_fit(array_map(fn($h) => [$h], $schools['study_hours']), $schools['score'],
    [['z' => array_fill(0, count($schools['score']), [1]), 'groups' => $schools['school']]]);
mcompare($lmer['coefficients'], $matrix['coefficients'], 'lmer equals stats_lmm_fit', 1e-12, 1e-12);
mcheck($lmer['method'] === 'ML' && $lmer['random'][0]['label'] === '1 | school' && $lmer['random'][0]['group_labels'] === $matrix['random'][0]['group_labels'], 'lmer metadata');
$panel = $fixture['datasets']['panel'];
$glmer = @stats_glmer('yb ~ x + (1 | subject)', $panel);
mcheck($glmer['family'] === 'binomial' && $glmer['random'][0]['group'] === 'subject' && $glmer['n_dropped'] === 2, 'glmer metadata');
foreach ([
    ['no random term', fn() => stats_lmer('score ~ study_hours', $schools), 1001, 'random-effect term'],
    ['glmer slopes', fn() => stats_glmer('yb ~ x + (x | subject)', $panel), 1007, 'exactly one random intercept'],
    ['glmer two terms', fn() => stats_glmer('yb ~ x + (1 | subject) + (1 | item)', $panel), 1007, 'exactly one random intercept'],
    ['glmer family', fn() => stats_glmer('yb ~ x + (1 | subject)', $panel, 'poisson'), 1007, 'binomial'],
    ['glmer link', fn() => stats_glmer('yb ~ x + (1 | subject)', $panel, link: 'probit'), 1007, 'logit'],
    ['lmer optimizer', fn() => stats_lmer('y ~ x + (1 | subject)', $panel, optimizer: 'nloptwrap'), 1007, 'optimizer'],
    ['lmer raise', fn() => stats_lmer('y ~ x + (1 | subject)', $panel, missing: 'raise'), 1001, "missing='raise'"],
] as [$label, $call, $code, $fragment]) {
    $thrown = null;
    try { @$call(); } catch (Throwable $e) { $thrown = $e; }
    mcheck($thrown !== null && $thrown->getCode() === $code && str_contains($thrown->getMessage(), $fragment), "$label: " . ($thrown?->getMessage() ?? 'no exception'));
}
$warnings = [];
set_error_handler(function (int $severity, string $message) use (&$warnings): bool { $warnings[] = $message; return true; });
try { $r = stats_lmer('y ~ x + (1 | subject)', $panel); } finally { restore_error_handler(); }
mcheck(count($warnings) === 1 && str_contains($warnings[0], 'stats_lmer dropped 5 of 384') && $r['dropped_rows'] === [4, 39, 76, 210, 299], 'lmer deletion warning');
echo "PHP formula mixed-model tests passed ({$fits['lmer']} lmer and {$fits['glmer']} glmer lme4 fits; $comparisons comparisons)\n";
