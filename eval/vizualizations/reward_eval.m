% =========================================================================
% Reward 1-5 kiertekelese az eval_scenes.py meresebol.
%
% A meres: 5 modell x 10 palya x 5 kor = 250 bejaras. Adott (palya, kor)
% parban minden modell ugyanazt a seedet kapta, ezert a modellek PAROSITVA
% hasonlithatok ossze: ugyanazon a forgalmon mert kulonbseg.
%
% Osszesites (mint a CARLA Leaderboardon):
%   1. palyankent az 5 kor osszesitve (osszes esemeny / osszes km stb.),
%   2. a 10 palya egyszeru atlaga - minden palya egyforma sullyal.
% A tablazatban a +/- a palyak kozti szoras: ez a SZORODAS (mennyire
% egyenletes a modell kulonbozo palyakon), NEM konfidenciaintervallum,
% ezert a modellek osszehasonlitasara nem hasznalhato [Patterson 2024].
%
% Osszehasonlitas: a 4 tervezett lepes (R1->R2, ..., R4->R5) parositott
% kulonbsege. 95%-os CI retegzett bootstrappal: palyankent a 5 kort
% visszatevessel ujramintavetelezzuk, a ket modellnel UGYANAZOKAT a korokat
% (parositas), es ugyanugy osszesitunk [Agarwal 2021]. Ha a CI nem
% tartalmazza a 0-t, a kulonbseg szignifikans. A 4 osszehasonlitasra
% metrikankent Holm-korrekcio. Ellenorzeskeppen parositott t-proba is a 50
% (palya, kor) paron [Colas 2019].
%
% A CSV hosszu formatumu: a "row" oszlop donti el, mi a sor (scene = bejaras
% osszesito, fail/hit/overtake/attempt/stuck = esemenyek).
% =========================================================================

clear; clc; close all;

% A script sajat helyehez kepest: a legfrissebb eredmenyfajl, es a dolgozat
% abrakonyvtara.
here        = fileparts(mfilename('fullpath'));
results_dir = fullfile(here, '..', 'results');
fig_dir     = fullfile(here, '..', '..', 'tdk', 'pictures', 'eval');
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

scenes   = T(T.row == "scene",   :);
attempts = T(T.row == "attempt", :);

% Bejarasonkenti megtett ut [km] es ido [h]. A visszateves teleportja nem
% szamit bele (az env.distance_traveled-bol jon az avg_speed_kmh).
scenes.km = scenes.avg_speed_kmh / 3.6 .* scenes.time_s / 1000;
scenes.h  = scenes.time_s / 3600;

rw   = unique(scenes.reward);
n    = numel(rw);
lbl  = "R" + string(rw);
scn  = unique(scenes.scene);
reps = unique(scenes.rep);
nR   = numel(scn);
nP   = numel(reps);

fprintf('Fajl: %s\n', d(newest).name);
fprintf('  bejaras: %d  (%d modell x %d palya x %d kor)\n\n', height(scenes), n, nR, nP);

% =========================================================================
% Bejarasonkenti osszegek egy [modell x palya x kor] tombben. Minden
% metrika egy hanyados: skala * szamlalo / nevezo. Palyan belul a szamlalot
% es a nevezot kulon osszegezzuk az 5 korre, igy egy rovidre leallt
% bejaras nem torzit, es a bootstrap ugyanigy osszesit.
% =========================================================================
met_id = {'reset_km', 'hit_km', 'legal_km', 'speed', ...
          'lane', 'stuck_km', 'jerk', 'hit_pass', ...
          'success_pct', 'missed_km', 'illegal_km', 'illegal_hit_pct', 'illegal_pct'};
met_tx = {'Visszatevés / km', 'Járműütközés / km', 'Szabályos előzés / km', 'Átlagsebesség [km/h]', ...
          'Sávközépen [%]', 'Beragadás [s/km]', 'Kormányváltozás / lépés', 'Ütközés / elhagyott autó [%]', ...
          'Sikeres előzési kísérlet [%]', 'Kihagyott előzés / km', 'Szabálytalan elhaladás / km', ...
          'Ütközés szabálytalannál [%]', 'Szabálytalan elhaladás [%]'};
met_up = [-1, -1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1];   % +1: a nagyobb a jobb
nM = numel(met_id);

NUM = cell(1, nM); DEN = cell(1, nM);
for m = 1:nM
    NUM{m} = nan(n, nR, nP); DEN{m} = nan(n, nR, nP);
end
SCL = [1, 1, 1, 1, 100, 1, 1, 100, 100, 1, 1, 100, 100];

% Elhaladasok bejarasonkent az overtake sorokbol: osszes, szabalytalan,
% es ezek kozul hany autot utott el (az elso erintes, barmikor).
overtakes = T(T.row == "overtake", :);
PASS = zeros(n, nR, nP); ILL = zeros(n, nR, nP); ILLHIT = zeros(n, nR, nP);
for r = 1:height(overtakes)
    o = overtakes(r, :);
    i = find(rw == o.reward); j = find(scn == o.scene); k = find(reps == o.rep);
    PASS(i, j, k)   = PASS(i, j, k) + 1;
    ILL(i, j, k)    = ILL(i, j, k) + (o.legal == 0);
    ILLHIT(i, j, k) = ILLHIT(i, j, k) + (o.legal == 0 && o.hit == 1);
end

% Kihagyott elozes: beragadas (legalabb 5 s ugyanazon auto mogott), ahol az
% ido nagyobb reszeben szaggatott volt a vonal es szabad a bal sav.
stucks = T(T.row == "stuck", :);
MISS = zeros(n, nR, nP);
for r = 1:height(stucks)
    o = stucks(r, :);
    i = find(rw == o.reward); j = find(scn == o.scene); k = find(reps == o.rep);
    MISS(i, j, k) = MISS(i, j, k) + (o.stuck_reason == "could_overtake");
end

for r = 1:height(scenes)
    s = scenes(r, :);
    i = find(rw == s.reward); j = find(scn == s.scene); k = find(reps == s.rep);
    NUM{1}(i, j, k) = s.resets;                      DEN{1}(i, j, k) = s.km;
    NUM{2}(i, j, k) = s.hit_cars;                    DEN{2}(i, j, k) = s.km;
    NUM{3}(i, j, k) = s.legal_overtakes;             DEN{3}(i, j, k) = s.km;
    NUM{4}(i, j, k) = s.km;                          DEN{4}(i, j, k) = s.h;
    NUM{5}(i, j, k) = s.lane_center_pct / 100 * s.time_s;   DEN{5}(i, j, k) = s.time_s;
    NUM{6}(i, j, k) = s.stuck_s;                     DEN{6}(i, j, k) = s.km;
    NUM{7}(i, j, k) = s.steer_jerk * s.time_s;       DEN{7}(i, j, k) = s.time_s;
    NUM{8}(i, j, k) = s.hit_cars;                    DEN{8}(i, j, k) = s.overtakes;
    NUM{9}(i, j, k) = s.overtake_success;            DEN{9}(i, j, k) = s.overtake_attempts;
    % Legalabb 5 s ugyanazon auto mogott, es az ido nagyobb reszeben elozhetett volna.
    NUM{10}(i, j, k) = MISS(i, j, k);                DEN{10}(i, j, k) = s.km;
    NUM{11}(i, j, k) = ILL(i, j, k);                 DEN{11}(i, j, k) = s.km;
    NUM{12}(i, j, k) = ILLHIT(i, j, k);              DEN{12}(i, j, k) = ILL(i, j, k);
    NUM{13}(i, j, k) = ILL(i, j, k);                 DEN{13}(i, j, k) = PASS(i, j, k);
end

% Palyankenti ertek [modell x palya], es a 10 palya atlaga / szorasa.
V = cell(1, nM); MU = nan(n, nM); SD = nan(n, nM);
for m = 1:nM
    V{m}     = SCL(m) * sum(NUM{m}, 3) ./ sum(DEN{m}, 3);
    MU(:, m) = mean(V{m}, 2, 'omitnan');
    SD(:, m) = std(V{m}, 0, 2, 'omitnan');
end

fprintf('Atlag +/- palyak kozti szoras (szorodas, nem CI):\n');
fprintf('%-30s', ''); fprintf('%18s', lbl); fprintf('\n');
for m = 1:nM
    fprintf('%-30s', met_id{m});
    for i = 1:n
        fprintf('%10.3f +/-%5.2f', MU(i, m), SD(i, m));
    end
    fprintf('\n');
end

% =========================================================================
% Parositott osszehasonlitas: retegzett bootstrap. Palyankent 5 kort
% huzunk visszatevessel; ugyanaz az index mindket modellnel es minden
% metrikanal, igy a parositas megmarad.
% =========================================================================
B = 10000;
rng(0);
I   = randi(nP, nR, nP, B);                     % [palya x huzas x bootstrap]
lin = repmat((1:nR)', 1, nP, B) + (I - 1) * nR; % linearis index egy [palya x kor] matrixba
nC  = n - 1;                                    % 4 tervezett osszehasonlitas
DELTA = nan(nC, nM); LO = nan(nC, nM); HI = nan(nC, nM);
PB = nan(nC, nM); PT = nan(nC, nM);
for m = 1:nM
    BV = nan(n, nR, B);                         % bootstrap palyaertekek
    for i = 1:n
        num = squeeze(NUM{m}(i, :, :));
        den = squeeze(DEN{m}(i, :, :));
        BV(i, :, :) = SCL(m) * sum(num(lin), 2) ./ sum(den(lin), 2);
    end
    for c = 1:nC
        DELTA(c, m) = mean(V{m}(c + 1, :) - V{m}(c, :), 'omitnan');
        db = squeeze(mean(BV(c + 1, :, :) - BV(c, :, :), 2, 'omitnan'));
        LO(c, m) = prctile(db, 2.5);
        HI(c, m) = prctile(db, 97.5);
        PB(c, m) = min(1, 2 * min(mean(db <= 0), mean(db >= 0)));
        % Ellenorzes: parositott t-proba a bejarasonkenti ertekeken (50 par).
        a = SCL(m) * NUM{m}(c, :, :) ./ DEN{m}(c, :, :);
        b = SCL(m) * NUM{m}(c + 1, :, :) ./ DEN{m}(c + 1, :, :);
        ok = isfinite(a(:)) & isfinite(b(:));
        [~, PT(c, m)] = ttest(b(ok), a(ok));
    end
end

% Holm-korrekcio metrikankent a 4 osszehasonlitasra.
PH = nan(nC, nM);
for m = 1:nM
    [ps, ord] = sort(PB(:, m));
    adj = min(1, (nC:-1:1)' .* ps);
    adj = cummax(adj);
    PH(ord, m) = adj;
end

fprintf('\nParositott kulonbsegek (uj - regi), 95%% bootstrap CI, Holm-korrigalt p:\n');
for m = 1:nM
    fprintf('%s\n', met_id{m});
    for c = 1:nC
        sig = '';
        if PH(c, m) < 0.05, sig = '  *'; end
        fprintf('  %s->%s  %+9.3f  [%+9.3f, %+9.3f]  p_boot=%.4f  p_Holm=%.4f  p_t=%.4f%s\n', ...
                lbl(c), lbl(c + 1), DELTA(c, m), LO(c, m), HI(c, m), PB(c, m), PH(c, m), PT(c, m), sig);
    end
end

% =========================================================================
% Kiegeszito leirok: route completion, a bejarasok vege, a visszatevesek
% oka (palyankent / km, majd atlag), az elozesi kiserletek kimenetele.
% =========================================================================
fprintf('\nRoute completion es lezaras:\n');
for i = 1:n
    s = scenes(scenes.reward == rw(i), :);
    fprintf('  %s  RC %.1f%%  celba ert %d/%d  10 visszateves miatt leallt %d\n', lbl(i), ...
            mean(s.completion_pct), sum(s.reached_b), height(s), sum(s.end_reason == "Too many resets"));
end

rs_id = {'off_track', 'stopped', 'too_fast'};
rs_tx = {'letért az útról', 'megállt', 'túl gyors'};
RS = nan(n, 3);
for i = 1:n
    for q = 1:3
        v = nan(nR, 1);
        for j = 1:nR
            s = scenes(scenes.reward == rw(i) & scenes.scene == scn(j), :);
            v(j) = sum(s.(rs_id{q})) / sum(s.km);
        end
        RS(i, q) = mean(v);
    end
end
fprintf('\nVisszatevesek oka [1/km] (palyaatlag):\n');
for i = 1:n
    fprintf('  %s  off %.3f  stop %.3f  fast %.3f\n', lbl(i), RS(i, 1), RS(i, 2), RS(i, 3));
end

kinds  = ["success", "aborted", "hit", "solid_line_return"];
kn_tx  = {'sikeres', 'feladta', 'ütközött', 'záróvonalon vissza', 'egyéb'};
A = zeros(n, numel(kinds) + 1); nA = zeros(n, 1); AKM = zeros(n, 1);
for i = 1:n
    a = attempts(attempts.reward == rw(i), :);
    nA(i) = height(a);
    AKM(i) = height(a) / sum(scenes.km(scenes.reward == rw(i)));
    for q = 1:numel(kinds)
        A(i, q) = 100 * sum(a.attempt_outcome == kinds(q)) / height(a);
    end
    A(i, end) = 100 - sum(A(i, 1:end-1));
end
fprintf('\nElozesi kiserletek (osszes bejaras):\n');
for i = 1:n
    fprintf('  %s  n=%d (%.2f/km)  siker %.1f  feladta %.1f  utkozott %.1f  zarovonal %.1f  egyeb %.1f\n', ...
            lbl(i), nA(i), AKM(i), A(i, :));
end

% Szabalytalan elhaladasok oka, kilometerenkent (palyaatlag), es az
% utkozesi arany szabalyos vs szabalytalan elhaladasnal (osszesitve).
il_id = ["right_side", "not_allowed", "solid_line_return"];
il_tx = {'jobbról', 'kísérleten kívül', 'záróvonalon vissza', 'egyéb'};
IL = zeros(n, numel(il_id) + 1);
fprintf('\nSzabalytalan elhaladasok oka [1/km] es utkozesi arany:\n');
for i = 1:n
    o = overtakes(overtakes.reward == rw(i), :);
    for j = 1:nR
        s  = scenes(scenes.reward == rw(i) & scenes.scene == scn(j), :);
        oj = o(o.scene == scn(j) & o.legal == 0, :);
        for q = 1:numel(il_id)
            IL(i, q) = IL(i, q) + sum(oj.illegal_reason == il_id(q)) / sum(s.km) / nR;
        end
        IL(i, end) = IL(i, end) + sum(~ismember(oj.illegal_reason, il_id)) / sum(s.km) / nR;
    end
    lg = o(o.legal == 1, :); il = o(o.legal == 0, :);
    fprintf('  %s  jobb %.2f  kivul %.2f  zaro %.2f  egyeb %.2f | szabalyos %d (utk %.1f%%)  szabalytalan %d (utk %.1f%%)\n', ...
            lbl(i), IL(i, :), height(lg), 100 * mean(lg.hit == 1), height(il), 100 * mean(il.hit == 1));
end

% LaTeX-sorok metrikankent: ertek +/- palyak kozti szoras, es a kulonbseg
% az elozo modellhez kepest a 95%-os CI-vel (* = Holm utan szignifikans).
fmt = {'%.2f', '%.2f', '%.2f', '%.1f', '%.1f', '%.1f', '%.3f', '%.1f', '%.1f', '%.2f', '%.2f', '%.1f', '%.1f'};
for m = 1:nM
    fprintf('\nLaTeX - %s:\n', met_id{m});
    for i = 1:n
        fprintf('    %s & %s\\,$\\pm$\\,%s', lbl(i), strrep(sprintf(fmt{m}, MU(i, m)), '.', '{,}'), ...
                strrep(sprintf(fmt{m}, SD(i, m)), '.', '{,}'));
        if i == 1
            fprintf(' & -- \\\\\n');
        else
            c = i - 1; star = '';
            if PH(c, m) < 0.05, star = '^{*}'; end
            fprintf(' & $%s%s$ {\\scriptsize[$%s$; $%s$]} \\\\\n', ...
                    strrep(sprintf(['%+' fmt{m}(2:end)], DELTA(c, m)), '.', '{,}'), star, ...
                    strrep(sprintf(['%+' fmt{m}(2:end)], LO(c, m)), '.', '{,}'), ...
                    strrep(sprintf(['%+' fmt{m}(2:end)], HI(c, m)), '.', '{,}'));
        end
    end
end

% =========================================================================
% Abrak. Szinek: a modellek identitasat a vizszintes tengely adja, ezert
% egy semleges szin eleg; a kiemeles (szignifikans) egy akcentusszin.
% =========================================================================
ink   = [0.04 0.04 0.04];
muted = [0.32 0.32 0.31];
grid_ = [0.80 0.80 0.78];
acc   = hex2rgb('#2a78d6');
cat5  = hex2rgb(["#1baf7a", "#eda100", "#e34948", "#6250d6", "#c3c2b7"]);
cat3  = hex2rgb(["#2a78d6", "#eb6834", "#1baf7a"]);
set(groot, 'defaultAxesFontSize', 9, 'defaultAxesXColor', muted, 'defaultAxesYColor', muted, ...
    'defaultAxesGridColor', grid_, 'defaultAxesGridAlpha', 1, 'defaultAxesBox', 'off', ...
    'defaultTextColor', ink, 'defaultAxesTitleFontWeight', 'normal');

% 3. abra - hogyan romlik el: az elozesi kiserletek kimenetele, es a
% visszatevesek oka kilometerenkent.
f3 = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 18 7]);
tl = tiledlayout(f3, 1, 2, 'TileSpacing', 'loose', 'Padding', 'compact');
ax = nexttile(tl);
b = bar(ax, A, 'stacked', 'BarWidth', 0.6, 'EdgeColor', 'w', 'LineWidth', 1);
for q = 1:size(A, 2), b(q).FaceColor = cat5(q, :); end
% A darabszam nelkul a szazalekok felrevezetok: a tengelyfeliratba kerul.
ax.XTickLabel = lbl + "\newline" + compose('%d', nA); ax.XTickLabelRotation = 0;
ylim(ax, [0 100]);
grid(ax, 'on'); ax.XGrid = 'off';
ylabel(ax, 'Előzési kísérletek [%]');
legend(ax, kn_tx, 'Location', 'eastoutside', 'Box', 'off');
ax = nexttile(tl);
b = bar(ax, RS, 'stacked', 'BarWidth', 0.6, 'EdgeColor', 'w', 'LineWidth', 1);
for q = 1:3, b(q).FaceColor = cat3(q, :); end
ax.XTickLabel = lbl; grid(ax, 'on'); ax.XGrid = 'off';
ylabel(ax, 'Visszatevés / km');
legend(ax, rs_tx, 'Location', 'eastoutside', 'Box', 'off');
exportgraphics(f3, fullfile(fig_dir, 'outcomes.pdf'), 'ContentType', 'vector');

% 4. abra - amikor nem szabadott volna elozni: a szabalytalan elhaladasok
% oka kilometerenkent, es hogy ezeknel hanyszor lett utkozes a szabalyos
% elhaladasokhoz kepest.
HR = zeros(n, 2);
for i = 1:n
    o = overtakes(overtakes.reward == rw(i), :);
    HR(i, 1) = 100 * mean(o.hit(o.legal == 1) == 1);
    HR(i, 2) = 100 * mean(o.hit(o.legal == 0) == 1);
end
f4 = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 18 7]);
tl = tiledlayout(f4, 1, 2, 'TileSpacing', 'loose', 'Padding', 'compact');
ax = nexttile(tl);
b = bar(ax, IL, 'stacked', 'BarWidth', 0.6, 'EdgeColor', 'w', 'LineWidth', 1);
ilcol = [cat3; cat5(end, :)];
for q = 1:size(IL, 2), b(q).FaceColor = ilcol(q, :); end
ax.XTickLabel = lbl; grid(ax, 'on'); ax.XGrid = 'off';
ylabel(ax, 'Szabálytalan elhaladás / km');
legend(ax, il_tx, 'Location', 'eastoutside', 'Box', 'off');
ax = nexttile(tl);
b = bar(ax, HR, 'BarWidth', 0.8, 'EdgeColor', 'w', 'LineWidth', 1);
b(1).FaceColor = muted; b(2).FaceColor = acc;
ax.XTickLabel = lbl; grid(ax, 'on'); ax.XGrid = 'off';
ylabel(ax, 'Elütött autó [%]');
legend(ax, {'szabályos', 'szabálytalan'}, 'Location', 'eastoutside', 'Box', 'off');
exportgraphics(f4, fullfile(fig_dir, 'illegal.pdf'), 'ContentType', 'vector');

% 5-6. abra - mind az 5 modell egymas mellett, utvonaltipusonkent. Sebesseg:
% a tipus 25 bejarasanak atlaga. Menetido: utvonalankent a celba ert
% bejarasok atlagos ideje, ezek osszege tipusonkent. Elozesek: a tipus
% osszes bejarasat osszesitve.
gt    = ["highway", "city_to_highway"];
gt_tx = {'autópálya', 'városból autópályára'};
SPD = nan(n, 2); TIM = nan(n, 2); LEG = nan(n, 2); SUC = nan(n, 2);
SJ = nan(n, 2); LC = nan(n, 2); HIT = nan(n, 2); RES = nan(n, 2); NP = nan(n, 2);
for i = 1:n
    for g = 1:2
        s = scenes(scenes.reward == rw(i) & scenes.group == gt(g), :);
        SPD(i, g) = mean(s.avg_speed_kmh);
        SJ(i, g)  = mean(s.steer_jerk);
        LC(i, g)  = mean(s.lane_center_pct);
        HIT(i, g) = sum(s.hit_cars);
        RES(i, g) = sum(s.resets);
        r = s(s.reached_b == 1, :);
        [~, ~, k] = unique(r.scene);
        TIM(i, g) = sum(accumarray(k, r.time_s, [], @mean));
        SUC(i, g) = 100 * sum(s.overtake_success) / sum(s.overtake_attempts);
        o = overtakes(overtakes.reward == rw(i) & overtakes.group == gt(g), :);
        LEG(i, g) = 100 * mean(o.legal == 1);
        NP(i, g)  = height(o);
    end
end
fprintf('\nUtvonaltipusonkent (autopalya | varosbol autopalyara):\n');
for i = 1:n
    fprintf(['  %s  seb %.1f | %.1f  ido %.0f | %.0f  szab %.1f | %.1f  sik %.1f | %.1f' ...
             '  korm %.4f | %.4f  savk %.1f | %.1f  utk %d | %d  vissza %d | %d  elhal %d | %d\n'], ...
            lbl(i), SPD(i, :), TIM(i, :), LEG(i, :), SUC(i, :), SJ(i, :), LC(i, :), HIT(i, :), RES(i, :), NP(i, :));
end

f5 = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 18 7]);
tl = tiledlayout(f5, 1, 2, 'TileSpacing', 'loose', 'Padding', 'compact');
ax = nexttile(tl);
b = bar(ax, SPD, 'BarWidth', 0.8, 'EdgeColor', 'w', 'LineWidth', 1);
for q = 1:2, b(q).FaceColor = cat3(q, :); end
ax.XTickLabel = lbl; grid(ax, 'on'); ax.XGrid = 'off';
ylabel(ax, 'Átlagsebesség [km/h]');
legend(ax, gt_tx, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
ax = nexttile(tl);
b = bar(ax, TIM, 'stacked', 'BarWidth', 0.6, 'EdgeColor', 'w', 'LineWidth', 1);
for q = 1:2, b(q).FaceColor = cat3(q, :); end
text(ax, 1:n, sum(TIM, 2), compose('%d s', round(sum(TIM, 2))), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'Color', ink);
ylim(ax, [0, 1.12 * max(sum(TIM, 2))]);
ax.XTickLabel = lbl; grid(ax, 'on'); ax.XGrid = 'off';
ylabel(ax, 'Menetidő [s]');
legend(ax, gt_tx, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
exportgraphics(f5, fullfile(fig_dir, 'eval_speed.pdf'), 'ContentType', 'vector');

f6 = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 18 7]);
tl = tiledlayout(f6, 1, 2, 'TileSpacing', 'loose', 'Padding', 'compact');
ax = nexttile(tl);
b = bar(ax, LEG, 'BarWidth', 0.8, 'EdgeColor', 'w', 'LineWidth', 1);
for q = 1:2, b(q).FaceColor = cat3(q, :); end
ax.XTickLabel = lbl; ylim(ax, [0 100]); grid(ax, 'on'); ax.XGrid = 'off';
ylabel(ax, 'Szabályos elhaladás [%]');
legend(ax, gt_tx, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
ax = nexttile(tl);
b = bar(ax, SUC, 'BarWidth', 0.8, 'EdgeColor', 'w', 'LineWidth', 1);
for q = 1:2, b(q).FaceColor = cat3(q, :); end
ax.XTickLabel = lbl; ylim(ax, [0 100]); grid(ax, 'on'); ax.XGrid = 'off';
ylabel(ax, 'Sikeres előzési kísérlet [%]');
legend(ax, gt_tx, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
exportgraphics(f6, fullfile(fig_dir, 'eval_overtakes.pdf'), 'ContentType', 'vector');

% 7. abra - kormanyzas es savtartas utvonaltipusonkent.
f7 = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 18 7]);
tl = tiledlayout(f7, 1, 2, 'TileSpacing', 'loose', 'Padding', 'compact');
ax = nexttile(tl);
b = bar(ax, SJ, 'BarWidth', 0.8, 'EdgeColor', 'w', 'LineWidth', 1);
for q = 1:2, b(q).FaceColor = cat3(q, :); end
ax.XTickLabel = lbl; grid(ax, 'on'); ax.XGrid = 'off';
ylabel(ax, 'Kormányváltozás / lépés');
legend(ax, gt_tx, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
ax = nexttile(tl);
b = bar(ax, LC, 'BarWidth', 0.8, 'EdgeColor', 'w', 'LineWidth', 1);
for q = 1:2, b(q).FaceColor = cat3(q, :); end
ax.XTickLabel = lbl; ylim(ax, [0 100]); grid(ax, 'on'); ax.XGrid = 'off';
ylabel(ax, 'Sávközépen [%]');
legend(ax, gt_tx, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
exportgraphics(f7, fullfile(fig_dir, 'eval_steer.pdf'), 'ContentType', 'vector');

% 8. abra - jarmuutkozesek es visszatevesek szama utvonaltipusonkent, a
% tipus 25 bejarasat osszesitve.
f8 = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 18 7]);
tl = tiledlayout(f8, 1, 2, 'TileSpacing', 'loose', 'Padding', 'compact');
ax = nexttile(tl);
b = bar(ax, HIT, 'stacked', 'BarWidth', 0.6, 'EdgeColor', 'w', 'LineWidth', 1);
for q = 1:2, b(q).FaceColor = cat3(q, :); end
text(ax, 1:n, sum(HIT, 2), compose('%d', sum(HIT, 2)), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'Color', ink);
ylim(ax, [0, 1.12 * max(sum(HIT, 2))]);
ax.XTickLabel = lbl; grid(ax, 'on'); ax.XGrid = 'off';
ylabel(ax, 'Elütött jármű [db]');
legend(ax, gt_tx, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
ax = nexttile(tl);
b = bar(ax, RES, 'stacked', 'BarWidth', 0.6, 'EdgeColor', 'w', 'LineWidth', 1);
for q = 1:2, b(q).FaceColor = cat3(q, :); end
text(ax, 1:n, sum(RES, 2), compose('%d', sum(RES, 2)), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'Color', ink);
ylim(ax, [0, 1.12 * max(sum(RES, 2))]);
ax.XTickLabel = lbl; grid(ax, 'on'); ax.XGrid = 'off';
ylabel(ax, 'Visszatevés [db]');
legend(ax, gt_tx, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
exportgraphics(f8, fullfile(fig_dir, 'eval_safety.pdf'), 'ContentType', 'vector');

% 9. abra - a celba nem ert bejarasok: hol tortentek a visszatevesek a
% terkepen. Balra a megallasok, jobbra a leteresek kornyeke; szurke: az
% osszes utvonal, sotetebb: az erintett utvonal, a pontok modellenkent.
routes = readtable(fullfile(results_dir, 'routes.csv'), 'TextType', 'string');
fails  = T(T.row == "fail", :);
nf     = scenes(scenes.reached_b == 0, :);
mcol   = hex2rgb(["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"]);
fprintf('\nCelba nem ert bejarasok:\n');
for r = 1:height(nf)
    d = fails(fails.reward == nf.reward(r) & fails.scene == nf.scene(r) & fails.rep == nf.rep(r), :);
    fprintf('  %s  %-15s kor %d  %.1f%%  %.2f-%.2f km  letert %d  megallt %d\n', lbl(rw == nf.reward(r)), ...
            nf.scene(r), nf.rep(r) + 1, nf.completion_pct(r), min(d.route_m) / 1000, max(d.route_m) / 1000, ...
            sum(d.reason == "Off-track"), sum(d.reason == "Vehicle stopped"));
end

f9 = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 18 8.5]);
tl = tiledlayout(f9, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
kinds_f = ["Vehicle stopped", "Off-track"];
box_f   = {[20 290 -215 -90], [-400 20 -280 0]};
tt_f    = {'Megállás egy álló jármű mögött', 'Letérés az útról'};
mk_f    = {'o', 'x'};
for p = 1:2
    ax = nexttile(tl); hold(ax, 'on');
    for s = unique(routes.scene)'
        rr = routes(routes.scene == s, :);
        plot(ax, rr.x, rr.y, '-', 'Color', grid_, 'LineWidth', 2.5);
    end
    sel = nf(ismember(nf.scene, unique(fails.scene(fails.reason == kinds_f(p)))), :);
    for s = unique(sel.scene)'
        rr = routes(routes.scene == s, :);
        plot(ax, rr.x, rr.y, '-', 'Color', muted, 'LineWidth', 1.2);
    end
    h = gobjects(0); hl = strings(0);
    for i = 1:n
        % Csak a celba nem ert bejarasok visszatevesei.
        d = fails(fails.reward == rw(i) & fails.reason == kinds_f(p), :);
        d = d(ismember(string(d.reward) + d.scene + d.rep, string(nf.reward) + nf.scene + nf.rep), :);
        bx = box_f{p};
        d = d(d.x >= bx(1) & d.x <= bx(2) & d.y >= bx(3) & d.y <= bx(4), :);
        if isempty(d), continue; end
        h(end + 1) = plot(ax, d.x, d.y, mk_f{p}, 'Color', mcol(i, :), 'MarkerFaceColor', 'none', ...
                          'MarkerSize', 7, 'LineWidth', 1.5);
        hl(end + 1) = lbl(i);
    end
    % Utvonalnev a pontcsoportok melle; az egymashoz kozeli csoportok
    % (ugyanott akadt el ket utvonal) egy kozos feliratot kapnak.
    bx = box_f{p};
    lx = []; ly = []; lt = strings(0);
    [~, ord_f] = sort(nf.scene);
    for r = ord_f'
        d = fails(fails.reward == nf.reward(r) & fails.scene == nf.scene(r) & fails.rep == nf.rep(r) & ...
                  fails.reason == kinds_f(p), :);
        d = d(d.x >= bx(1) & d.x <= bx(2) & d.y >= bx(3) & d.y <= bx(4), :);
        if isempty(d), continue; end
        nm = extractAfter(nf.scene(r), "city_highway_");
        % A leteresek egy hosszu szakaszon szorodnak, ott nagyobb a sugar.
        k = find(hypot(lx - mean(d.x), ly - mean(d.y)) < 80 + 320 * (p == 2), 1);
        if isempty(k)
            lx(end + 1) = mean(d.x); ly(end + 1) = mean(d.y);
            lt(end + 1) = "V→A " + nm;
        elseif ~contains(lt(k), nm)
            lt(k) = lt(k) + ", " + nm;
        end
    end
    text(ax, lx, ly - 12, lt, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 8, 'Color', ink);
    axis(ax, 'equal'); axis(ax, box_f{p}); ax.YDir = 'reverse';
    ax.XTick = []; ax.YTick = []; ax.XColor = 'none'; ax.YColor = 'none';
    title(ax, tt_f{p}, 'FontSize', 10);
    legend(ax, h, hl, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
end
exportgraphics(f9, fullfile(fig_dir, 'eval_fail.pdf'), 'ContentType', 'vector');

fprintf('\nAbrak: %s\n', fig_dir);
