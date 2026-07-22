#!/bin/bash
set -e

case "${ENV}" in
  3s5z_vs_3s6z)
    # Super hard asymmetric (3S+5Z vs 3S+6Z).
    MAP="3s5z_vs_3s6z"
    DEFAULT_TMAX=2050000
    ;;
  corridor)
    # Super hard: 6 Zealots in narrow corridor vs 24 Zerglings.
    MAP="corridor"
    DEFAULT_TMAX=2050000
    ;;
  MMM2)
    # Super hard: 1 Medivac + 2 Marauders + 7 Marines vs stronger enemy.
    MAP="MMM2"
    DEFAULT_TMAX=2050000
    ;;
  *)
    echo "Unknown ENV label: ${ENV}" >&2
    exit 1
    ;;
esac

TMAX="${TMAX_OVERRIDE:-${DEFAULT_TMAX}}"
TEST_INTERVAL="${TEST_INTERVAL_OVERRIDE:-50000}"

python src/main.py --config=custom --env-config=smaclite \
    with env_args.map_name="${MAP}" \
    t_max=${TMAX} \
    test_interval=${TEST_INTERVAL} \
    test_nepisode=32 \
    seed=${SEED:-42} \
    common_reward=True \
    use_cuda=True \
    epsilon_anneal_time=50000
