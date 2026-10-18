set -u -o pipefail

REPEATS="${1:-5}"
# Masodik argumentum: scene-fajl (pl. eval/scenes_city.json csak a varosi
# utvonalakhoz). Ha nincs megadva, a teljes eval/scenes.json fut.
SCENES="${2:-eval/scenes.json}"
# Harmadik argumentum: egy korabbi teljes eval CSV. Ebbol azok a sorok
# kerulnek at az uj CSV-be, amelyek csoportja (highway / city_to_highway)
# most nem futott - igy csak a varosi resz ujrafuttatasa utan is egy teljes
# CSV lesz.
BASE_CSV="${3:-}"

# A script helyehez kepest, hogy mindegy legyen, honnan inditjuk.
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$EVAL_DIR")"
cd "$REPO_DIR" || exit 1

LOG_DIR="$EVAL_DIR/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
CSV="$EVAL_DIR/results/eval_$TS.csv"

MODELS=(
    tensorboard/SAC_reward_1/model_final.zip
    tensorboard/SAC_reward_2_part2/model_final.zip
    tensorboard/SAC_reward_3/model_final.zip
    tensorboard/SAC_reward_4/model_final.zip
    tensorboard/SAC_reward_5/model_final.zip
)

# Hianyzo modell: inkabb alljunk meg itt, mint orak mulva egy felkesz CSV-vel.
for m in "${MODELS[@]}"; do
    if [[ ! -f "$m" ]]; then
        echo "[HIBA] Nincs meg a modell: $m" >&2
        exit 1
    fi
done
if [[ -n "$BASE_CSV" && ! -f "$BASE_CSV" ]]; then
    echo "[HIBA] Nincs meg a korabbi CSV: $BASE_CSV" >&2
    exit 1
fi

echo "[INFO] Korok:  $REPEATS"
echo "[INFO] Scene:  $SCENES"
echo "[INFO] Modell: ${#MODELS[@]} db, ket CARLA szerveren parhuzamosan"
echo "[INFO] CSV:    $CSV"
echo

# Ket folyamat parhuzamosan, mindegyik sajat CARLA szerverrel (2000 / 2010,
# traffic manager 8000 / 8010) es sajat reszCSV-vel: az elso harom modell az
# egyiken, a masik ketto a masikon. A seed scene-enkent fix, igy az eredmeny
# ugyanaz, mintha egy folyamat futtatna. A kimenet a logba ES a kepernyore is
# megy, a kepernyon a port elotaggal.
conda run --no-capture-output -n carla_rl \
    python eval/eval_scenes.py --repeats "$REPEATS" --scenes "$SCENES" --port 2000 \
    --out "$EVAL_DIR/results/part_${TS}_2000.csv" --model "${MODELS[@]:0:3}" 2>&1 \
    | tee "$LOG_DIR/eval_${TS}_2000.log" | sed -u "s/^/[2000] /" &
PID_A=$!
# A ket szerver ne egyszerre induljon.
sleep 20
conda run --no-capture-output -n carla_rl \
    python eval/eval_scenes.py --repeats "$REPEATS" --scenes "$SCENES" --port 2010 \
    --out "$EVAL_DIR/results/part_${TS}_2010.csv" --model "${MODELS[@]:3}" 2>&1 \
    | tee "$LOG_DIR/eval_${TS}_2010.log" | sed -u "s/^/[2010] /" &
PID_B=$!
wait "$PID_A" "$PID_B"

# A reszCSV-k (es a korabbi CSV hianyzo csoportjai) egyetlen eval CSV-be,
# utana a reszCSV-k torlodnek.
python3 - "$CSV" "$BASE_CSV" "$EVAL_DIR/results/part_${TS}_2000.csv" "$EVAL_DIR/results/part_${TS}_2010.csv" <<'EOF'
import csv, os, sys
out, base, parts = sys.argv[1], sys.argv[2], [p for p in sys.argv[3:] if os.path.isfile(p)]
rows, cols = [], None
for p in parts:
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        cols = cols or r.fieldnames
        rows += list(r)
if base:
    with open(base, newline="") as f:
        r = csv.DictReader(f)
        cols = cols or r.fieldnames
        ran = {x["group"] for x in rows}
        rows = [x for x in r if x["group"] not in ran] + rows
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, cols)
    w.writeheader()
    w.writerows(rows)
for p in parts:
    os.remove(p)
print(f"[INFO] {len(rows)} sor -> {out}")
EOF

echo
echo "[INFO] Log:  $LOG_DIR/eval_${TS}_2000.log, $LOG_DIR/eval_${TS}_2010.log"
echo "[INFO] CSV:  $CSV"
# A python a szenzorok felszabaditasanal a KIIRAS UTAN is elszallhat
# (core dumped). Ilyenkor az eredmeny mar a CSV-ben van, ezert ezt nem
# tekintjuk sikertelen futasnak - a logban ott kell lennie az utolso scene-nek.
