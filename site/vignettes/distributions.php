<?php
// Vignette for phit.phpstats.org: Student's t against the normal distribution.
declare(strict_types=1);

use function Phit\{dist_normal_pdf_array, dist_normal_quantile, dist_normal_sf, dist_t_pdf_array, dist_t_quantile, dist_t_sf};

return [
    'inputs' => [
        'df' => ['type' => 'number', 'label' => 'Degrees of freedom', 'min' => 1, 'max' => 60, 'step' => 1, 'default' => 3],
    ],
    'run' => function (array $input, string $data): array {
        $df = (float) $input['df'];
        $xs = array_map(fn ($i) => round(-5 + $i * 0.05, 2), range(0, 200));
        $t = dist_t_pdf_array($xs, $df);
        $normal = dist_normal_pdf_array($xs);
        $curves = [];
        foreach ($xs as $i => $x) {
            $curves[] = ['x' => $x, 'density' => $t[$i], 'distribution' => 't, df = ' . (int) $df];
            $curves[] = ['x' => $x, 'density' => $normal[$i], 'distribution' => 'Normal'];
        }
        $rows = [];
        foreach ([1.645, 1.96, 2.576, 3.291] as $q) {
            $rows[] = ['|x| >' => $q, 't' => 2 * dist_t_sf($q, $df), 'normal' => 2 * dist_normal_sf($q)];
        }

        return [
            ['type' => 'text', 'markdown' => "Student's t with " . (int) $df . ' degrees of freedom has heavier tails than the normal. '
                . 'Every value here comes from the native distribution functions, which are checked against 60-digit references.'],
            ['type' => 'code', 'chain' => "use function Phit\\{dist_t_pdf_array, dist_t_sf};\n\n"
                . "dist_t_pdf_array(\$xs, " . (int) $df . ");  // one native call for the whole grid\n"
                . "2 * dist_t_sf(1.96, " . (int) $df . ");      // two-sided tail probability", 'pipe' => null],
            ['type' => 'metrics', 'values' => ['t critical value (95%)' => dist_t_quantile(0.975, $df), 'Normal critical value' => dist_normal_quantile(0.975)]],
            ['type' => 'chart', 'title' => 'Densities', 'spec' => [
                '$schema' => 'https://vega.github.io/schema/vega-lite/v5.json',
                'width' => 'container', 'height' => 300, 'data' => ['values' => $curves], 'mark' => 'line',
                'encoding' => ['x' => ['field' => 'x', 'type' => 'quantitative'], 'y' => ['field' => 'density', 'type' => 'quantitative'],
                    'color' => ['field' => 'distribution', 'type' => 'nominal', 'title' => null]],
            ]],
            ['type' => 'table', 'title' => 'Two-sided tail probabilities', 'columns' => ['|x| >', 't', 'normal'], 'rows' => $rows],
        ];
    },
];
