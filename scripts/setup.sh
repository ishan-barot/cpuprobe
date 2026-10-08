#!/usr/bin/env bash
# gets a native linux box ready for benchmarking. safe to run more than once.
# everything here resets on reboot.
set -u

echo "== cpu governor"
if command -v cpupower >/dev/null; then
    sudo cpupower frequency-set -g performance >/dev/null && echo "set to performance"
else
    echo "cpupower not found (fine in wsl, on native run: sudo apt install linux-tools-\$(uname -r))"
fi

echo "== perf counter access"
sudo sysctl -q kernel.perf_event_paranoid=1 && echo "perf_event_paranoid = $(cat /proc/sys/kernel/perf_event_paranoid)"

echo "== can we see the pmu?"
if command -v perf >/dev/null && perf stat -e cycles true 2>&1 | grep -q "not supported"; then
    echo "no, cycles shows not supported (vm or wsl)"
elif command -v perf >/dev/null; then
    echo "yes, counters work"
else
    echo "perf isnt installed, cant check"
fi

echo "== reminders"
echo "close the browser, discord and games, they add noise"
echo "pin runs to one core, the default --cpu 2 already does that"
