set -u -o pipefail

REPEATS="${1:-5}"

# A script helyehez kepest, hogy mindegy legyen, honnan inditjuk.
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$EVAL_DIR")"
cd "$REPO_DIR" || exit 1

LOG_DIR="$EVAL_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/eval_$(date +%Y%m%d-%H%M%S).log"

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

echo "[INFO] Korok:  $REPEATS"
echo "[INFO] Modell: ${#MODELS[@]} db"
echo "[INFO] Epizod: $((REPEATS * ${#MODELS[@]} * 10))"
echo "[INFO] Log:    $LOG"
echo

# A kimenet a kepernyore ES a logba is megy (tee), hogy kozben is lehessen
# kovetni. A PIPESTATUS a pythone, nem a tee-e.
conda run --no-capture-output -n carla_rl \
    python eval/eval_scenes.py --repeats "$REPEATS" --model "${MODELS[@]}" 2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"

echo
echo "[INFO] Log:  $LOG"
# A legfrissebb CSV: ezt irta ez a futas.
CSV="$(ls -t "$EVAL_DIR/results"/eval_*.csv 2>/dev/null | head -1)"
[[ -n "$CSV" ]] && echo "[INFO] CSV:  $CSV"

# A python a szenzorok felszabaditasanal a KIIRAS UTAN is elszallhat
# (core dumped). Ilyenkor az eredmeny mar a CSV-ben van, ezert ezt csak
# jelezzuk, nem tekintjuk sikertelen futasnak.
if [[ "$status" -ne 0 ]]; then
    echo "[FIGYELEM] A python $status kilepesi koddal zarult."
    echo "           Ha a logban ott az utolso scene sora, az adat megvan."
fi
exit "$status"
