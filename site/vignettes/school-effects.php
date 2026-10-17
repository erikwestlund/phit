<?php
// Vignette for phit.phpstats.org: school random intercepts with their conditional uncertainty.
declare(strict_types=1);

use function Phit\{fit_lmer, read_csv};

return [
    'inputs' => [
        'method' => ['type' => 'radio', 'label' => 'Estimation', 'options' => ['reml' => 'REML', 'ml' => 'Maximum likelihood'], 'default' => 'reml'],
    ],
    'run' => function (array $input, string $data): array {
        $reml = $input['method'] === 'reml';
        $csv = read_csv($data . '/schools.csv');
        $schools = array_combine($csv['names'], array_column($csv['columns'], 'values'));

        $fit = fit_lmer('score ~ study_hours + (1 | school)', $schools, reml: $reml);
        $term = $fit['random'][0];
        $effects = [];
        foreach ($term['group_labels'] as $g => $label) {
            $effect = $term['effects'][$g][0];
            $sd = sqrt($term['effects_cond_var'][$g][0][0]);
            $effects[] = ['school' => (string) $label, 'effect' => $effect, 'lower' => $effect - 1.96 * $sd, 'upper' => $effect + 1.96 * $sd];
        }
        usort($effects, fn ($a, $b) => $a['effect'] <=> $b['effect']);

        return [
            ['type' => 'text', 'markdown' => '`fit_lmer()` fits a random intercept for each school, as `lme4::lmer()` does, by '
                . ($reml ? 'REML' : 'maximum likelihood') . ". Each school's effect is shown with a 95% interval from its "
                . 'conditional variance, which R reports as `ranef(fit, condVar = TRUE)`.'],
            ['type' => 'code', 'chain' => "use function Phit\\fit_lmer;\n\n"
                . "\$fit = fit_lmer('score ~ study_hours + (1 | school)', \$schools, reml: " . ($reml ? 'true' : 'false') . ");\n"
                . "\$fit['random'][0]['effects'];          // one intercept per school\n"
                . "\$fit['random'][0]['effects_cond_var']; // and its conditional variance", 'pipe' => null],
            ['type' => 'metrics', 'values' => ['School SD' => sqrt($term['covariance'][0][0]), 'Residual σ' => $fit['sigma'],
                'Schools' => count($term['group_labels']), 'Students' => $fit['n_observations']]],
            ['type' => 'model', 'title' => 'Fixed effects', 'result' => $fit],
            ['type' => 'chart', 'title' => 'School intercepts with 95% conditional intervals', 'spec' => [
                '$schema' => 'https://vega.github.io/schema/vega-lite/v5.json',
                'width' => 'container', 'height' => max(240, 16 * count($effects)),
                'data' => ['values' => $effects],
                'encoding' => ['y' => ['field' => 'school', 'type' => 'nominal', 'sort' => null, 'title' => null]],
                'layer' => [
                    ['mark' => 'rule', 'encoding' => ['x' => ['field' => 'lower', 'type' => 'quantitative', 'title' => 'Effect on score'], 'x2' => ['field' => 'upper']]],
                    ['mark' => ['type' => 'point', 'filled' => true], 'encoding' => ['x' => ['field' => 'effect', 'type' => 'quantitative']]],
                ],
            ]],
        ];
    },
];
