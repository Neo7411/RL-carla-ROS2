% =========================================================================
% Reward 1-5 kiertekelese az eval_scenes.py meresebol.
%
% A meres: 5 modell x 10 palya x 5 kor = 250 epizod. Minden modell ugyanazt
% az 50 (palya, seed) part futotta, ezert az osszehasonlitas tisztessegesl.
%
% Ket szamot kozlunk metrikankent:
%   atlag    - a modell teljesitmenye az osszes epizodon
%   szoras   - mennyire ingadozik epizodrol epizodra. Ha a szoras nagyobb,
%              mint a modellek kozti kulonbseg, akkor a kulonbseg nem
%              megbizhato: egy masik 5 kor mas sorrendet adhatna.
%
% A CSV hosszu formatumu: a "row" oszlop donti el, mi a sor (scene = epizod
% osszesito, fail/hit/overtake/attempt/stuck = esemenyek), az adott
% sortipushoz nem tartozo oszlopok uresek.
% =========================================================================

clear; clc; close all;

% A script sajat helyehez kepest, a legfrissebb eredmenyfajl.
results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
d = dir(fullfile(results_dir, 'eval_*.csv'));
if isempty(d)
    error('Nincs eredmenyfajl itt: %s', results_dir);
end
[~, newest] = max([d.datenum]);
csv_path = fullfile(results_dir, d(newest).name);

% Minden szoveges oszlop string, a tobbi double: igy az ures cellak NaN-ok.
% Az "end" oszlopbol a MATLAB "xEnd"-et csinal (kulcsszo), atnevezzuk.
opts = detectImportOptions(csv_path);
text_cols = {'model', 'group', 'scene', 'row', 'xEnd', 'reason', 'collision_with', ...
             'actor_type', 'ego_side', 'illegal_reason', 'attempt_outcome', 'stuck_reason'};
opts = setvartype(opts, text_cols, 'string');
opts = setvartype(opts, setdiff(opts.VariableNames, text_cols), 'double');
opts.EmptyLineRule = 'skip';

warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
T = readtable(csv_path, opts);
T = renamevars(T, 'xEnd', 'end_reason');

% A modellnevbol a reward szama (SAC_reward_3_model_final -> 3).
T.reward = double(extractBetween(T.model, 'SAC_reward_', '_'));

% Sortipusok kulon tablaba.
scenes    = T(T.row == "scene",    :);
fails     = T(T.row == "fail",     :);
hits      = T(T.row == "hit",      :);
overtakes = T(T.row == "overtake", :);
attempts  = T(T.row == "attempt",  :);
stucks    = T(T.row == "stuck",    :);

% Epizodonkenti megtett ut [km]: az atlagsebessegbol es az idobol.
scenes.km = scenes.avg_speed_kmh / 3.6 .* scenes.time_s / 1000;

rw  = unique(scenes.reward);
n   = numel(rw);
lbl = "R" + string(rw);
scn = unique(scenes.scene);   % a 10 palya

fprintf('Fajl: %s\n', d(newest).name);
fprintf('  epizod: %d  (%d modell x %d palya x %d kor)\n', height(scenes), ...
        n, numel(unique(scenes.scene)), numel(unique(scenes.rep)));
fprintf('  esemenyek: %d elozes, %d kiserlet, %d beragadas, %d utkozes, %d visszateves\n\n', ...
        height(overtakes), height(attempts), height(stucks), height(hits), height(fails));

% =========================================================================
% Osszesito tabla. Ketlepcsos atlagolas: eloszor palyankent (az 5 kor
% egybevetve), utana a 10 palya atlaga. Igy minden palya egyforma sullyal
% szamit, nem a hosszuak dominalnak.
% A szoras a 10 palya kozti ingadozas: mennyire egyenletes a modell.
% =========================================================================
M = table(rw, 'VariableNames', {'reward'});
for i = 1:n
    s = scenes(scenes.reward == rw(i), :);

    M.epizod(i)   = height(s);
    M.km(i)       = sum(s.km);

    % Palyankenti ertekek: eloszor az 5 kor egy palyan, utana a 10 palya.
    % Igy minden palya egyforma sullyal szamit, fuggetlenul attol, hogy
    % 1 km vagy 6 km hosszu. (Ha mindent egybeontenenk, a hosszu palyak
    % dominalnak: a legrovidebb 1.05 km, a leghosszabb 6.09 km.)
    % A per-km es a %-os ertekek palyan belul osszesitve (osszes esemeny /
    % osszes km), igy egy rovidre leallt epizod nem torzit.
    P = nan(numel(scn), 6);   % palya x metrika
    for j = 1:numel(scn)
        q = s(s.scene == scn(j), :);
        P(j, 1) = mean(q.avg_speed_kmh);
        P(j, 2) = sum(q.legal_overtakes) / sum(q.km);
        P(j, 3) = 100 * sum(q.hit_cars) / sum(q.overtakes);
        P(j, 4) = sum(q.resets) / sum(q.km);
        P(j, 5) = mean(q.lane_center_pct);
        P(j, 6) = mean(q.stuck_s);
    end

    % Az atlag a 10 palya atlaga, a szoras a palyak kozti ingadozas: ez
    % mutatja, mennyire egyenletes a modell a kulonbozo palyakon.
    M.speed(i)    = mean(P(:, 1));   M.speed_sd(i) = std(P(:, 1));
    M.legal_km(i) = mean(P(:, 2));   M.legal_sd(i) = std(P(:, 2));
    M.hit_pct(i)  = mean(P(:, 3));   M.hit_sd(i)   = std(P(:, 3));
    M.reset_km(i) = mean(P(:, 4));   M.reset_sd(i) = std(P(:, 4));
    M.lane(i)     = mean(P(:, 5));   M.lane_sd(i)  = std(P(:, 5));
    M.stuck(i)    = mean(P(:, 6));   M.stuck_sd(i) = std(P(:, 6));

    % Elozesi kiserletek: hany volt, es hany szazaleka sikerult.
    a = attempts(attempts.reward == rw(i), :);
    M.attempts(i) = height(a);
    M.success(i)  = 100 * sum(a.attempt_outcome == "success") / height(a);
end

disp('Osszesites (palyankent atlagolva, majd a 10 palya atlaga):');
disp(M(:, {'reward', 'epizod', 'km', 'speed', 'legal_km', 'hit_pct', ...
           'reset_km', 'lane', 'stuck', 'attempts', 'success'}));

% A fo metrikak nev szerint, es hogy a nagyobb-e a jobb.
met    = {'speed',  'legal_km', 'hit_pct', 'reset_km', 'lane',  'stuck'};
met_sd = {'speed_sd', 'legal_sd', 'hit_sd', 'reset_sd', 'lane_sd', 'stuck_sd'};
met_tx = {'Atlagsebesseg [km/h]', 'Szabalyos elozes / km', ...
          'Utkozes / elhagyott auto [%]', 'Visszateves / km', ...
          'Sav kozepen toltott ido [%]', 'Beragadas epizodonkent [s]'};
met_up = [+1, +1, -1, -1, +1, -1];   % +1: a nagyobb a jobb

fprintf('\nAtlag +/- szoras (a 10 palya atlaga; a szoras a palyak kozti\ningadozas: mennyire egyenletes a modell kulonbozo palyakon):\n');
for k = 1:numel(met)
    fprintf('  %-30s', met_tx{k});
    for i = 1:n
        fprintf('  %s: %6.2f +/-%5.2f', lbl(i), M.(met{k})(i), M.(met_sd{k})(i));
    end
    fprintf('\n');
end

% Ha a modellek kozti teljes kulonbseg kisebb, mint az atlagos szoras,
% akkor a sorrend nem megbizhato - ezt kiirjuk, hogy ne lehessen
% veletlenul tulertelmezni.
fprintf('\nMegbizhatosag (a modellek kozti kulonbseg vs az ingadozas):\n');
for k = 1:numel(met)
    spread = max(M.(met{k})) - min(M.(met{k}));
    noise  = mean(M.(met_sd{k}));
    if spread > noise
        verdict = 'a kulonbseg nagyobb az ingadozasnal';
    else
        verdict = 'AZ INGADOZAS NAGYOBB - a sorrend bizonytalan';
    end
    fprintf('  %-30s  kulonbseg %6.2f, ingadozas %6.2f  -> %s\n', ...
            met_tx{k}, spread, noise, verdict);
end

% =========================================================================
% 1. abra - a hat fo metrika, hibasavval (atlag +/- szoras).
% A zold oszlop a legjobb ertek. Ahol a hibasavok atfednek, ott a
% kulonbseg nem meggyozo.
% =========================================================================
figure('Name', 'Fo metrikak', 'Color', 'w', 'Position', [100 100 1000 620]);
for k = 1:numel(met)
    subplot(2, 3, k);
    hold on; grid on;
    v  = M.(met{k});
    sd = M.(met_sd{k});
    b = bar(v, 'FaceColor', 'flat');
    b.CData = repmat([0.65 0.65 0.68], n, 1);
    if met_up(k) > 0
        [~, best] = max(v);
    else
        [~, best] = min(v);
    end
    b.CData(best, :) = [0.20 0.65 0.32];
    errorbar(1:n, v, sd, 'k', 'LineStyle', 'none', 'LineWidth', 1.1, 'CapSize', 8);
    set(gca, 'XTick', 1:n, 'XTickLabel', lbl);
    title(met_tx{k}, 'FontSize', 10);
    ylim([0, max(v + sd) * 1.15]);
    hold off;
end
sgtitle(sprintf('Reward 1-5 (%d epizod/modell, hibasav = palyak kozti szoras)', M.epizod(1)));

% =========================================================================
% 2. abra - sebesseg vs utkozes. A jobb also sarok a nyero: gyors es
% biztonsagos. A hibasavok mindket iranyban az ingadozast mutatjak.
% =========================================================================
figure('Name', 'Sebesseg vs utkozes', 'Color', 'w', 'Position', [120 120 660 520]);
hold on; grid on;
col = lines(n);
for i = 1:n
    errorbar(M.speed(i), M.hit_pct(i), M.hit_sd(i), M.hit_sd(i), ...
             M.speed_sd(i), M.speed_sd(i), 'o', 'MarkerSize', 10, ...
             'MarkerFaceColor', col(i, :), 'MarkerEdgeColor', 'k', ...
             'Color', [0.6 0.6 0.6], 'LineWidth', 1.0);
    text(M.speed(i) + 0.3, M.hit_pct(i) + 0.6, lbl(i), 'FontSize', 12, 'FontWeight', 'bold');
end
xlabel('Atlagsebesseg [km/h]');
ylabel('Utkozes / elhagyott auto [%]');
title('Sebesseg es biztonsag (hibasav = palyak kozti szoras)');
hold off;

% =========================================================================
% 3. abra - koronkenti bontas. Ha egy modell minden korben hasonloan
% teljesit, a vonala lapos; ha ugral, akkor a modell (vagy a meres)
% ingadozik, es egyetlen kor eredmenye felrevezetne.
% =========================================================================
reps = unique(scenes.rep);
figure('Name', 'Koronkent', 'Color', 'w', 'Position', [140 140 900 420]);
for k = 1:2
    subplot(1, 2, k);
    hold on; grid on;
    for i = 1:n
        y = nan(numel(reps), 1);
        for j = 1:numel(reps)
            s = scenes(scenes.reward == rw(i) & scenes.rep == reps(j), :);
            if k == 1
                y(j) = sum(s.legal_overtakes) / sum(s.km);
            else
                y(j) = mean(s.avg_speed_kmh);
            end
        end
        plot(reps + 1, y, '-o', 'LineWidth', 1.4, 'Color', col(i, :), ...
             'MarkerFaceColor', col(i, :), 'DisplayName', lbl(i));
    end
    xlabel('kor'); xticks(reps + 1);
    if k == 1
        ylabel('Szabalyos elozes / km');
        title('Elozes koronkent');
        legend('Location', 'best');
    else
        ylabel('Atlagsebesseg [km/h]');
        title('Sebesseg koronkent');
    end
    hold off;
end
sgtitle('Koronkenti eredmeny: mennyire stabil a modellek sorrendje');

% =========================================================================
% 4. abra - az elozesi kiserletek kimenetele. Nem csak az latszik, hany
% sikerul, hanem az is, hogyan romlik el: feladja, utkozik, vagy
% zarovonalon sorol vissza.
% =========================================================================
kinds  = ["success", "aborted", "hit", "solid_line_return"];
knames = {'sikeres', 'feladta', 'utkozott', 'zarovonalon vissza', 'egyeb'};
kcol   = [0.20 0.65 0.32; 0.95 0.72 0.20; 0.84 0.25 0.22; 0.45 0.45 0.80; 0.80 0.80 0.80];
A = zeros(n, numel(kinds) + 1);
for i = 1:n
    a = attempts(attempts.reward == rw(i), :);
    for j = 1:numel(kinds)
        A(i, j) = 100 * sum(a.attempt_outcome == kinds(j)) / height(a);
    end
    A(i, end) = 100 - sum(A(i, 1:end-1));
end

figure('Name', 'Elozesi kiserletek', 'Color', 'w', 'Position', [160 160 760 500]);
b = bar(A, 'stacked');
for j = 1:size(A, 2)
    b(j).FaceColor = kcol(j, :);
end
set(gca, 'XTickLabel', lbl);
ylabel('Az elozesi kiserletek megoszlasa [%]');
ylim([0 112]); grid on;
legend(knames, 'Location', 'eastoutside');
title('Elozesi kiserletek kimenetele');
% A darabszam nelkul a szazalekok felrevezetok.
text(1:n, repmat(103, 1, n), compose('n=%d', M.attempts), ...
     'HorizontalAlignment', 'center', 'FontSize', 9);

% =========================================================================
% 5. abra - a ket palyatipus kulon. A varosi szakasz mas feladat, mint az
% autopalya, es az osszevont atlag elfedheti, ha egy modell csak az
% egyikben jo.
% =========================================================================
gnames = ["highway", "city_to_highway"];
G = zeros(n, 2); H = zeros(n, 2);
for i = 1:n
    for j = 1:2
        s = scenes(scenes.reward == rw(i) & scenes.group == gnames(j), :);
        G(i, j) = sum(s.legal_overtakes) / sum(s.km);
        H(i, j) = 100 * sum(s.hit_cars) / sum(s.overtakes);
    end
end

figure('Name', 'Palyatipusok', 'Color', 'w', 'Position', [180 180 900 420]);
subplot(1, 2, 1);
bar(G); set(gca, 'XTickLabel', lbl); grid on;
ylabel('Szabalyos elozes / km');
title('Elozes palyatipusonkent');
legend({'autopalya', 'varos -> autopalya'}, 'Location', 'northwest');

subplot(1, 2, 2);
bar(H); set(gca, 'XTickLabel', lbl); grid on;
ylabel('Utkozes / elhagyott auto [%]');
title('Utkozes palyatipusonkent');
legend({'autopalya', 'varos -> autopalya'}, 'Location', 'northwest');
sgtitle('A ket palyatipus kulon');
