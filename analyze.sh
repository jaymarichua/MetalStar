#!/usr/bin/env bash
#
# analyze_logs.sh
#
# Finds the newest *.txt file in the actor_log directory, then checks for
# spam or toxic lines. Adjust paths as needed.

# Adjust this path to wherever your actor logs are stored
LOG_DIR="experiments/test/actor_log"

echo "[INFO] Searching for the latest .txt log in: $LOG_DIR"

# Attempt to list all .txt files, sort by modified time descending,
# then pick the first (most recent) file.
LATEST_TXT_FILE=$(ls -t1 "$LOG_DIR"/*.txt 2>/dev/null | head -n 1)

# If none found, warn and exit
if [[ -z "$LATEST_TXT_FILE" ]]; then
    echo "[WARNING] No .txt logs found in $LOG_DIR"
    exit 1
fi

echo "[INFO] Found most recent .txt log file: $LATEST_TXT_FILE"

# Now we grep for lines that mention spam or toxic
echo
echo "[INFO] Searching for RollingRewardHackingMonitor lines..."
SPAM_LINES=$(grep "RollingRewardHackingMonitor" "$LATEST_TXT_FILE" || true)

echo "[INFO] Searching for ToxicStrategyMonitor lines..."
TOXIC_LINES=$(grep "ToxicStrategyMonitor" "$LATEST_TXT_FILE" || true)

# Display them (or do further processing as needed)
echo
echo "===== WARNING ! ====="
echo "SPAM DETECTION"
if [[ -n "$SPAM_LINES" ]]; then
    echo "$SPAM_LINES"
else
    echo "(none)"
fi

echo
echo "===== WARNING ! ====="
echo "TOXICITY DETECTION"
if [[ -n "$TOXIC_LINES" ]]; then
    echo "$TOXIC_LINES"
else
    echo "(none)"
fi
