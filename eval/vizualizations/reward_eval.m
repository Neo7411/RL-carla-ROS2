%% Reward fejlesztes lepesenkent: mit valtoztattam, es melyik mutato mozdult tole
% R1 -> R2 -> R3 -> R4 -> R5: minden lepes egy valtoztatas a rewardon. Minden
% modell ugyanazt a 10 scene-t vezeti (Town04, azonos route, forgalom es
% seed), igy egy lepes hatasa scene-enkent parositva merheto:
%   - Wilcoxon elojeles rang teszt a ket egymast koveto modell kozott,
%   - Holm-korrekcio a 4 lepesre (metrikankent),
%   - jobb / rosszabb scene: hany scene-ben mozdult a jo / rossz iranyba.
% Csak az ervenyes mutatok: menetido, atlagsebesseg, route teljesites, sav
% kozepe, kormany-rangatas, elutott autok, beavatkozasok. Az overtakes oszlop
% nem csak a szabalyos elozeseket szamolja, ezert kimarad.
% Kimenet (figs/): steps.csv, summary.csv, fig1_steps.pdf/png, fig2_scene_times.pdf/png

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
resDir = fullfile(here, '..', 'results');
outDir = fullfile(here, 'figs');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% results/ mappanev, rovid nev, mi valtozott az elozo modellhez kepest
models = {
    'SAC_reward_1_model_final',       'R1', 'alap: sávtartás + szabályos előzés'
    'SAC_reward_2_part2_model_final', 'R2', 'TTC-büntetés előzhető helyzetben is'
    'SAC_reward_3_model_final',       'R3', 'bátrabb előzés (TTC 7→3 s, nagyobb előzési jutalom)'
    'SAC_reward_4_model_final',       'R4', 'haladás-alapú jutalom a kézi tagok helyett'
    'SAC_reward_5_model_final',       'R5', 'potenciál-alapú formálás (PBRS) az előzésre'
};
nM = size(models, 1);
nSt = nM - 1;
short = models(:, 2)';

% Modellenkent rogzitett szin
C = [0.165 0.471 0.839     % #2a78d6
     0.922 0.408 0.204     % #eb6834
     0.106 0.686 0.478     % #1baf7a
     0.929 0.631 0.000     % #eda100
     0.910 0.482 0.643];   % #e87ba4

% oszlop, felirat, mertekegyseg, irany (+1: a nagyobb jobb, -1: a kisebb jobb)
mets = {
    'rel_time',        'Menetidő',               's',       -1
    'avg_speed_kmh',   'Átlagsebesség',          'km/h',    +1
    'completion_pct',  'Route teljesítés',       '%',       +1
    'lane_center_pct', 'Sávközépen töltött idő', '%',       +1
    'steer_jerk',      'Kormány-rángatás',       '1/lépés', -1
    'hits_per_km',     'Elütött autók',          'db/km',   -1
    'resets_per_km',   'Beavatkozások',          'db/km',   -1
};
nK = size(mets, 1);

%% Beolvasas: modellenkent a legutolso futas (idobelyeg mappa) scenes.csv-je
cols = {'time_s', 'avg_speed_kmh', 'completion_pct', 'lane_center_pct', 'steer_jerk', ...
        'hit_cars', 'resets', 'reached_b'};
for m = 1:nM
    runs = dir(fullfile(resDir, models{m, 1}, '2*'));
    runs = runs([runs.isdir]);
    if isempty(runs), error('Nincs futas: %s', models{m, 1}); end
    runDir = fullfile(runs(end).folder, runs(end).name);   % nev szerint rendezett -> a vege a legujabb
    s = readtable(fullfile(runDir, 'scenes.csv'), 'TextType', 'string', 'VariableNamingRule', 'preserve');
    if m == 1
        scenes = s.scene;
        nS = numel(scenes);
        for c = 1:numel(cols), X.(cols{c}) = nan(nS, nM); end
    end
    [~, idx] = ismember(scenes, s.scene);
    for c = 1:numel(cols), X.(cols{c})(:, m) = s.(cols{c})(idx); end
end
Done = X.reached_b == 1;
X.dist_km = X.avg_speed_kmh .* X.time_s / 3600;   % megtett ut, a visszatevesek nelkul
X.hits_per_km = X.hit_cars ./ X.dist_km;
X.resets_per_km = X.resets ./ X.dist_km;
% Menetido a scene leggyorsabb celba ert modelljehez kepest [%]: a scene-ek
% kulonbozo hosszuak, igy lesznek egy skalan. Ami nem ert celba, kimarad.
Tdone = X.time_s; Tdone(~Done) = NaN;
X.rel_time = 100 * Tdone ./ min(Tdone, [], 2);

% Osszesitett ertek modellenkent (mind a 10 scene): ut / ido, idovel sulyozott atlag
for m = 1:nM
    tw = X.time_s(:, m) / sum(X.time_s(:, m));
    Pm.avg_speed_kmh(m) = 3600 * sum(X.dist_km(:, m)) / sum(X.time_s(:, m));
    Pm.completion_pct(m) = mean(X.completion_pct(:, m));
    Pm.lane_center_pct(m) = sum(tw .* X.lane_center_pct(:, m));
    Pm.steer_jerk(m) = sum(tw .* X.steer_jerk(:, m));
    Pm.hits_per_km(m) = sum(X.hit_cars(:, m)) / sum(X.dist_km(:, m));
    Pm.resets_per_km(m) = sum(X.resets(:, m)) / sum(X.dist_km(:, m));
end

%% Lepesenkenti osszevetes
pS = nan(nK, nSt); pH = nan(nK, nSt); nBetter = pS; nWorse = pS; nPair = pS;
before = pS; after = pS;
for k = 1:nK
    col = mets{k, 1};
    for st = 1:nSt
        a = X.(col)(:, st); b = X.(col)(:, st + 1);
        ok = ~isnan(a) & ~isnan(b);
        d = mets{k, 4} * (b(ok) - a(ok));     % > 0: a jo iranyba mozdult
        nBetter(k, st) = sum(d > 0); nWorse(k, st) = sum(d < 0); nPair(k, st) = sum(ok);
        pS(k, st) = signrank(a(ok), b(ok));
        if col == "rel_time"
            % Menetido: az osszes ido azokon a scene-eken, ahol mindketto celba ert
            before(k, st) = sum(X.time_s(ok, st)); after(k, st) = sum(X.time_s(ok, st + 1));
        else
            before(k, st) = Pm.(col)(st); after(k, st) = Pm.(col)(st + 1);
        end
    end
    % Holm-Bonferroni a 4 lepesre
    [ps, o] = sort(pS(k, :));
    pH(k, o) = min(1, cummax((nSt - (0:nSt - 1)) .* ps));
end

% Tabla lepesenkent, azon belul metrikankent
Lepes = strings(0, 1); Valtoztatas = Lepes; Metrika = Lepes; Eredmeny = Lepes;
Elotte = []; Utana = []; Valtozas_pct = []; Jobb_scene = []; Rosszabb_scene = []; N_scene = []; P = []; P_Holm = [];
for st = 1:nSt
    for k = 1:nK
        Lepes(end + 1, 1) = sprintf('%s → %s', short{st}, short{st + 1});
        Valtoztatas(end + 1, 1) = models{st + 1, 3};
        Metrika(end + 1, 1) = sprintf('%s [%s]', mets{k, 2}, mets{k, 3});
        Elotte(end + 1, 1) = before(k, st); Utana(end + 1, 1) = after(k, st);
        Valtozas_pct(end + 1, 1) = 100 * (after(k, st) - before(k, st)) / before(k, st);
        Jobb_scene(end + 1, 1) = nBetter(k, st); Rosszabb_scene(end + 1, 1) = nWorse(k, st);
        N_scene(end + 1, 1) = nPair(k, st);
        P(end + 1, 1) = pS(k, st); P_Holm(end + 1, 1) = pH(k, st);
        if pH(k, st) >= 0.05
            Eredmeny(end + 1, 1) = "nem szignifikáns";
        elseif nBetter(k, st) > nWorse(k, st)
            Eredmeny(end + 1, 1) = "szignifikánsan javult";
        else
            Eredmeny(end + 1, 1) = "szignifikánsan romlott";
        end
    end
end
Steps = table(Lepes, Valtoztatas, Metrika, Elotte, Utana, Valtozas_pct, Jobb_scene, Rosszabb_scene, ...
              N_scene, P, P_Holm, Eredmeny);
writetable(Steps, fullfile(outDir, 'steps.csv'));

% Osszesito modellenkent
Summary = table(string(short'), sum(Done, 1)', Pm.completion_pct', sum(X.time_s, 1)', Pm.avg_speed_kmh', ...
                Pm.lane_center_pct', Pm.steer_jerk', sum(X.hit_cars, 1)', Pm.hits_per_km', ...
                sum(X.resets, 1)', Pm.resets_per_km', ...
                'VariableNames', {'Modell', 'Celba_ert', 'Teljesites_pct', 'Ossz_ido_s', 'Atlagsebesseg_kmh', ...
                                  'Savkozep_pct', 'Kormany_rangatas', 'Elutott_autok', 'Elutott_per_km', ...
                                  'Beavatkozasok', 'Beavatkozas_per_km'});
writetable(Summary, fullfile(outDir, 'summary.csv'));

%% Konzol
disp(Summary);
for st = 1:nSt
    fprintf('\n%s → %s: %s\n', short{st}, short{st + 1}, models{st + 1, 3});
    for k = 1:nK
        fprintf('   %-24s %9.4g → %-9.4g %-8s (%+6.1f%%)  jobb %2d / rosszabb %2d a %2d scene-bol  p=%.4f  pHolm=%.4f  %s\n', ...
                mets{k, 2}, before(k, st), after(k, st), mets{k, 3}, ...
                100 * (after(k, st) - before(k, st)) / before(k, st), nBetter(k, st), nWorse(k, st), ...
                nPair(k, st), pS(k, st), pH(k, st), Steps.Eredmeny((st - 1) * nK + k));
    end
end
fprintf('\n(Menetido: osszes ido azokon a scene-eken, ahol mindket modell celba ert.)\n');

%% 1. abra: a mutatok lepesrol lepesre, scene-enkent (szurke) es a median (fekete)
% * / ** a ket modell kozott: szignifikans valtozas (Holm p < 0.05 / 0.01).
panels = [1 2 4 5 6 7];          % a route teljesites szinte mindenhol 100%, kimarad
f1 = figure('Units', 'centimeters', 'Position', [2 2 16 11], 'Color', 'w'); theme(f1, 'light');
tl = tiledlayout(f1, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:numel(panels)
    k = panels(i);
    Y = X.(mets{k, 1});
    ax = nexttile(tl); hold(ax, 'on');
    for j = 1:nS
        hS = plot(ax, 1:nM, Y(j, :), '-', 'Color', [0.80 0.80 0.78], 'LineWidth', 0.75);
    end
    med = median(Y, 1, 'omitnan');
    hMed = plot(ax, 1:nM, med, '-', 'Color', [0.2 0.2 0.2], 'LineWidth', 2);
    scatter(ax, 1:nM, med, 45, C, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1);
    yl = ylim(ax); ytop = yl(2) + 0.06 * diff(yl);
    for st = 1:nSt
        if pH(k, st) < 0.01, txt = '**'; elseif pH(k, st) < 0.05, txt = '*'; else, txt = ''; end
        text(ax, st + 0.5, ytop, txt, 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    end
    ylim(ax, [yl(1), ytop + 0.06 * diff(yl)]);
    xlim(ax, [0.7 nM + 0.3]); xticks(ax, 1:nM); xticklabels(ax, short);
    if mets{k, 1} == "rel_time"
        ylabel(ax, 'a scene leggyorsabbjához [%]');
    else
        ylabel(ax, mets{k, 3});
    end
    if mets{k, 4} > 0, dirTxt = '(nagyobb = jobb)'; else, dirTxt = '(kisebb = jobb)'; end
    title(ax, mets{k, 2}, 'FontWeight', 'normal', 'FontSize', 9);
    subtitle(ax, dirTxt, 'FontSize', 7, 'Color', [0.32 0.32 0.31]);
    grid(ax, 'on'); ax.GridAlpha = 0.12; ax.Box = 'off'; ax.FontSize = 8;
    if i == 1
        legend(ax, [hS hMed], {'egy scene', 'medián'}, 'Location', 'northwest', 'Box', 'off', 'FontSize', 7);
    end
end
exportgraphics(f1, fullfile(outDir, 'fig1_steps.pdf'), 'ContentType', 'vector');
exportgraphics(f1, fullfile(outDir, 'fig1_steps.png'), 'Resolution', 200);

%% 2. abra: menetido scene-enkent, a leggyorsabb celba ert csillaggal
[tBest, iBest] = min(Tdone, [], 2);
f2 = figure('Units', 'centimeters', 'Position', [2 2 16 7.5], 'Color', 'w'); theme(f2, 'light');
ax = axes(f2); hold(ax, 'on');
b = bar(ax, X.time_s, 'grouped', 'BarWidth', 0.85, 'EdgeColor', 'none');
for m = 1:nM, b(m).FaceColor = C(m, :); end
xb = reshape([b.XEndPoints], nS, nM);
hBest = plot(ax, xb(sub2ind([nS nM], (1:nS)', iBest)), tBest + 25, 'p', 'MarkerSize', 7, ...
             'MarkerFaceColor', [0.2 0.2 0.2], 'MarkerEdgeColor', 'none');
hFail = plot(ax, xb(~Done), X.time_s(~Done) + 25, 'x', 'MarkerSize', 7, 'LineWidth', 1.5, 'Color', [0.2 0.2 0.2]);
ylabel(ax, 'Menetidő [s]');
xticks(ax, 1:nS); xticklabels(ax, strrep(scenes, '_', ' ')); xtickangle(ax, 30);
legend(ax, [b, hBest, hFail], [short, {'leggyorsabb', 'nem ért célba'}], ...
       'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off', 'FontSize', 8);
grid(ax, 'on'); ax.GridAlpha = 0.12; ax.XGrid = 'off'; ax.Box = 'off'; ax.FontSize = 8;
exportgraphics(f2, fullfile(outDir, 'fig2_scene_times.pdf'), 'ContentType', 'vector');
exportgraphics(f2, fullfile(outDir, 'fig2_scene_times.png'), 'Resolution', 200);

fprintf('\nKimenet: %s\n', outDir);
