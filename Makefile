TRACK ?= T1
GPU ?= 0
SEED ?= 0
MODE ?= clean
CKPT ?=
ENV_NAME ?= troncamp_env

.PHONY: check env install restore-paths configs collect process process-interact train train-interact eval eval-local watch submit structure

check:
	bash scripts/00_check_prereqs.sh

env:
	ENV_NAME=$(ENV_NAME) bash scripts/01_create_env.sh

install:
	ENV_NAME=$(ENV_NAME) bash scripts/02_install_robotwin_act.sh

restore-paths:
	bash scripts/03_restore_paths.sh

configs:
	bash scripts/04_prepare_task_configs.sh

collect:
	bash scripts/10_collect_one.sh $(TRACK) $(GPU)

process:
	bash scripts/20_process_one.sh $(TRACK)

process-interact:
	bash scripts/20_process_interact_one.sh $(TRACK)

train:
	bash scripts/30_train_one.sh $(TRACK) $(SEED) $(GPU)

train-interact:
	bash scripts/30_train_interact_one.sh $(TRACK) $(SEED) $(GPU)

eval:
	bash scripts/40_eval_one.sh $(TRACK) $(MODE) $(SEED) $(GPU)

eval-local:
	bash scripts/50_eval_local_one.sh $(TRACK)

watch:
	bash scripts/51_watch_rollout.sh $(TRACK) $(SEED)

submit:
	@if [ -n "$(CKPT)" ]; then \
		bash scripts/60_submit_one.sh $(TRACK) "$(CKPT)"; \
	else \
		bash scripts/60_submit_one.sh $(TRACK); \
	fi

structure:
	find . -maxdepth 3 -not -path './.git*' -not -path './.agents*' -not -path './.codex*' | sort
