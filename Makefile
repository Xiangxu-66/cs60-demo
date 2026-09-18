.DEFAULT_GOAL := help

.PHONY: help setup-mac setup-cuda docker-build train train-baseline train-debug \
        train-2000 train-5000 \
        evaluate benchmark benchmark-baseline \
        ablation-bottleneck ablation-depth ablation-loss ablation-all \
        test lint format export-tables clean \
		train-vrwkv-2000 train-dinov3-2000 train-dinov3-rwkv7-2000 train-dinov3-rwkv7-full train-dinov2-2000 \
		train-dinov3-cnn-skip-attn-160 train-lut-debug train-lut-full train-lut-cnn-full train-lut-cnn-160 \
		train-color-naf-debug train-color-naf-160 train-color-naf-full \
		train-color-naf-psnr-debug train-color-naf-psnr-160 train-color-naf-psnr-full \
		train-color-naf-base-v2-full train-color-naf-base-v2-160 \
		train-color-naf-film-v2-full train-color-naf-film-v2-160 \
		train-color-naf-base-v3-full train-color-naf-base-v3-160 train-color-naf-base-v3-debug \
		train-color-naf-base-v4-full train-color-naf-base-v4-160 train-color-naf-base-v4-debug \
		train-final docker-train

CONFIG_NAME ?= config
TORCH_EXTENSIONS_DIR ?= $(CURDIR)/outputs/torch_extensions
PYTHON ?= python
ARGS ?=

# Preferred workflow:
#   make train ARGS='family=color_naf_rwkv7 experiment_name=my_run'
# Keep CONFIG_NAME=config unless you intentionally switch the top-level Hydra entry point.
TRAIN := $(PYTHON) scripts/train.py --config-name $(CONFIG_NAME)
EVALUATE := $(PYTHON) scripts/evaluate.py
BENCHMARK := $(PYTHON) scripts/benchmark_efficiency.py
EXPORT_RESULTS := $(PYTHON) scripts/export_results.py

help:
	@printf "%s\n" \
	"RWKV Color Enhancement - Make targets" \
	"" \
	"Preferred workflow:" \
	"  make train ARGS='family=color_naf_rwkv7 experiment_name=my_run'" \
	"  make train ARGS='family=color_naf_rwkv7 ablation/color_naf_rwkv7/film=on experiment_name=my_run'" \
	"  make train ARGS='encoder=dinov3_b bottleneck=rwkv7_vision decoder=cnn experiment_name=my_run'" \
	"" \
	"Notes:" \
	"  - Leave CONFIG_NAME=config unless you are switching the top-level Hydra config." \
	"  - Use 'make -n <target>' to preview the expanded command without running it." \
	"  - Activate the conda env first: conda activate rwkv-color" \
	"" \
	"Common targets:" \
	"  make train                Run training with Hydra overrides passed through ARGS" \
	"  make train-final          Final paper run: 4500-img pool, no val, last ckpt (pass ARGS for arch/name)" \
	"  make train-debug          2-epoch smoke test" \
	"  make train-2000           Repo split run: first 2000 train, fixed 500 val/test" \
	"  make train-5000           Repo split run: 4000 train, fixed 500 val/test" \
	"  make benchmark            FLOPs + latency profiling" \
	"  make evaluate ARGS='--checkpoint ... --config ...'" \
	"  make test                 Run pytest from the active environment" \
	"  make lint                 Ruff + Black + isort checks" \
	"  make format               Auto-format Python files" \
	"" \
	"Historical experiment presets:" \
	"  make train-color-naf-base-v3-full" \
	"  make train-color-naf-film-v2-160" \
	"  make train-lut-cnn-full"

# ============ Environment ============
setup-mac:
	conda env create -f conda/env-mps.yml
	@echo "Activate: conda activate rwkv-color"

setup-cuda:
	conda env create -f conda/env-cuda.yml
	@echo "Activate: conda activate rwkv-color"

docker-build:
	docker build -t rwkv-color -f docker/Dockerfile .

# ============ Training ============
train:
	$(TRAIN) $(ARGS)

train-final:
	$(TRAIN) run_mode=final_4500_no_val $(ARGS)

train-baseline:
	$(TRAIN) experiment_name=dino_cnn_cnn $(ARGS)

train-debug:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(TRAIN) experiment_name=smoke_dino_cnn_cnn \
		training.max_epochs=2 training.accumulate_grad_batches=8 \
		training.batch_size_per_device=1 \
		data.train_subset_size=8 data.val_subset_size=4 data.test_subset_size=4 $(ARGS)

train-lut-debug:
	$(TRAIN) \
		encoder=dinov3_b bottleneck=rwkv_v9 decoder=cnn_lut \
		model.encoder_skip_layers='[2,5,8,11]' \
		experiment_name=smoke_lut \
		training.learning_rate=1e-4 training.weight_decay=1e-4 \
		training.early_stopping.patience=8 decoder.residual_scale=0.5 \
		training.max_epochs=2 training.accumulate_grad_batches=4 \
		training.batch_size_per_device=8 data.crop_size=480 \
		data.train_subset_size=8 data.val_subset_size=4 $(ARGS)

train-lut-full:
	$(TRAIN) \
		encoder=dinov3_b bottleneck=rwkv_v9 decoder=cnn_lut \
		model.encoder_skip_layers='[2,5,8,11]' \
		training.max_epochs=50 training.learning_rate=1e-4 training.weight_decay=1e-4 \
		training.early_stopping.patience=8 decoder.residual_scale=0.5 \
		experiment_name=lut_full $(ARGS)

train-lut-cnn-full:
	$(TRAIN) \
		encoder=dinov3_b +cnn_encoder=cnn_freq_gated bottleneck=rwkv7 decoder=cnn_lut_cnn \
		decoder.num_luts=5 decoder.lut_size=33 decoder.use_1d_lut=true \
		decoder.base_channels=64 decoder.num_upsample_blocks=4 decoder.residual_scale=0.5 \
		decoder.cnn_feature_channels='[64,64,64,64]' \
		training.max_epochs=50 training.learning_rate=1e-4 training.weight_decay=1e-4 \
		training.warmup_epochs=5 training.early_stopping.patience=8 \
		experiment_name=lut_cnn_full $(ARGS)

train-lut-cnn-160:
	$(TRAIN) \
		encoder=dinov3_b +cnn_encoder=cnn_freq_gated bottleneck=rwkv7 decoder=cnn_lut_cnn \
		decoder.num_luts=5 decoder.lut_size=33 decoder.use_1d_lut=true \
		decoder.base_channels=64 decoder.num_upsample_blocks=4 decoder.residual_scale=0.5 \
		decoder.cnn_feature_channels='[64,64,64,64]' \
		training.learning_rate=1e-4 training.weight_decay=1e-4 \
		training.warmup_epochs=5 \
		training.max_epochs=20 training.early_stopping.patience=5 \
		training.accumulate_grad_batches=8 training.batch_size_per_device=1 data.crop_size=256 \
		data.train_subset_size=160 \
		data.val_subset_size=20 data.test_subset_size=20 $(ARGS)

train-dinov3-cnn-skip-attn-160:
	$(TRAIN) experiment_name=dinov3_cnn_cnn_skip_attention_160 \
		encoder=dinov3_b bottleneck=cnn decoder=cnn_skip decoder.skip_fusion_mode=attention_gate \
		model.encoder_skip_layers='[2,5,8,11]' \
		training.max_epochs=20 training.early_stopping.patience=5 \
		training.batch_size_per_device=8 data.train_subset_size=160 \
		data.val_subset_size=20 data.test_subset_size=20 $(ARGS)

train-2000:
	$(TRAIN) run_mode=train_2000_fixed $(ARGS)

train-5000:
	$(TRAIN) run_mode=train_5000_fixed $(ARGS)

train-vrwkv-2000:
	TORCH_EXTENSIONS_DIR=$(TORCH_EXTENSIONS_DIR) $(TRAIN) ablation=a_loss_l1_only \
		experiment_name=vrwkv_2000img encoder=vrwkv_b \
		data.train_subset_size=2000 training.max_epochs=100 \
		training.early_stopping.patience=15 profile=true $(ARGS)

train-dinov3-2000:
	$(TRAIN) ablation=a_loss_l1_only \
		experiment_name=dinov3_2000img encoder=dinov3_b \
		data.train_subset_size=2000 training.max_epochs=100 \
		training.early_stopping.patience=15 profile=true $(ARGS)

train-dinov3-rwkv7-2000:
	$(TRAIN) ablation=a_loss_l1_only \
		experiment_name=dinov3_rwkv7_cnn_2000img encoder=dinov3_b bottleneck=rwkv7 decoder=cnn \
		data.train_subset_size=2000 training.max_epochs=100 \
		training.early_stopping.patience=15 profile=true $(ARGS)

train-dinov3-rwkv7-full:
	$(TRAIN) \
		experiment_name=dinov3_rwkv7_cnn_full encoder=dinov3_b bottleneck=rwkv7 decoder=cnn \
		training.max_epochs=100 training.early_stopping.patience=15 profile=true $(ARGS)

train-dinov2-2000:
	$(TRAIN) ablation=a_loss_l1_only \
		experiment_name=dinov2_2000img encoder=dinov2_b \
		data.train_subset_size=2000 training.max_epochs=100 \
		training.early_stopping.patience=15 data.crop_size=252 profile=true $(ARGS)

# ── base_v2 / film_v2 ──────────────────────────────────────────────────────
train-color-naf-base-v2-160:
	$(TRAIN) family=color_naf_rwkv7 \
		experiment_name=color_naf_base_v2_160img \
		data.train_subset_size=160 \
		data.val_subset_size=20 data.test_subset_size=20 $(ARGS)

train-color-naf-base-v2-full:
	$(TRAIN) family=color_naf_rwkv7 \
		experiment_name=color_naf_base_v2_full $(ARGS)

train-color-naf-film-v2-160:
	$(TRAIN) family=color_naf_rwkv7 \
		ablation/color_naf_rwkv7/film=on \
		experiment_name=color_naf_film_v2_160img \
		data.train_subset_size=160 \
		data.val_subset_size=20 data.test_subset_size=20 $(ARGS)

train-color-naf-film-v2-full:
	$(TRAIN) family=color_naf_rwkv7 \
		ablation/color_naf_rwkv7/film=on \
		experiment_name=color_naf_film_v2_full $(ARGS)

# ── base_v3: no ColorNAF skip (fusion-only hypothesis) ─────────────────────
train-color-naf-base-v3-debug:
	$(TRAIN) family=color_naf_rwkv7 \
		ablation/common/feature_skip=off \
		experiment_name=smoke_color_naf_base_v3 \
		training.max_epochs=2 training.accumulate_grad_batches=4 \
		training.batch_size_per_device=4 data.crop_size=256 \
		data.train_subset_size=8 data.val_subset_size=4 $(ARGS)

train-color-naf-base-v3-160:
	$(TRAIN) family=color_naf_rwkv7 \
		ablation/common/feature_skip=off \
		experiment_name=color_naf_base_v3_160img \
		training.batch_size_per_device=32 training.accumulate_grad_batches=1 \
		data.train_subset_size=160 \
		data.val_subset_size=20 data.test_subset_size=20 $(ARGS)

train-color-naf-base-v3-full:
	$(TRAIN) family=color_naf_rwkv7 \
		ablation/common/feature_skip=off \
		experiment_name=color_naf_base_v3_full \
		training.batch_size_per_device=32 training.accumulate_grad_batches=1 $(ARGS)

# ── base_v4: base_v2 + img_skip=true (re-validate under fixed optimizer) ───
train-color-naf-base-v4-debug:
	$(TRAIN) family=color_naf_rwkv7 \
		ablation/common/img_skip=on \
		experiment_name=smoke_color_naf_base_v4 \
		training.max_epochs=2 training.accumulate_grad_batches=4 \
		training.batch_size_per_device=4 data.crop_size=256 \
		data.train_subset_size=8 data.val_subset_size=4 $(ARGS)

train-color-naf-base-v4-160:
	$(TRAIN) family=color_naf_rwkv7 \
		ablation/common/img_skip=on \
		experiment_name=color_naf_base_v4_160img \
		training.batch_size_per_device=32 training.accumulate_grad_batches=1 \
		data.train_subset_size=160 \
		data.val_subset_size=20 data.test_subset_size=20 $(ARGS)

train-color-naf-base-v4-full:
	$(TRAIN) family=color_naf_rwkv7 \
		ablation/common/img_skip=on \
		experiment_name=color_naf_base_v4_full \
		training.batch_size_per_device=32 training.accumulate_grad_batches=1 $(ARGS)

docker-train:
	docker run --gpus all \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/outputs:/workspace/outputs \
		rwkv-color $(ARGS)

# ============ Evaluation ============
evaluate:
	@if [ -z "$(strip $(ARGS))" ]; then \
		echo "Usage: make evaluate ARGS='--checkpoint outputs/checkpoints/<name>.ckpt --config outputs/logs/<experiment>/config/config.yaml'"; \
		exit 2; \
	fi
	$(EVALUATE) $(ARGS)

benchmark:
	$(BENCHMARK) $(ARGS)

benchmark-baseline:
	$(BENCHMARK) --bottleneck cnn \
		--encoder-dim 768 --patch-size 14 $(ARGS)

# ============ Ablation batch runs ============
ablation-bottleneck:
	@for bn in none cnn; do \
		echo "=== Running bottleneck=$$bn ===" ; \
		$(TRAIN) encoder=dinov2_b decoder=cnn bottleneck=$$bn \
			experiment_name=A_bn_$$bn $(ARGS) ; \
	done

ablation-depth:
	@for d in 2 4 6; do \
		echo "=== Running CNN depth=$$d ===" ; \
		$(TRAIN) encoder=dinov2_b decoder=cnn bottleneck=cnn \
			bottleneck.num_layers=$$d experiment_name=A_cnn_depth_$$d $(ARGS) ; \
	done

ablation-loss:
	$(TRAIN) experiment_name=A_loss_stage1 $(ARGS)
	$(TRAIN) ablation=a_loss_l1_only experiment_name=A_loss_l1 $(ARGS)
	$(TRAIN) ablation=a_loss_l1_ssim experiment_name=A_loss_l1_ssim $(ARGS)

ablation-all:
	CONFIG_NAME=$(CONFIG_NAME) PYTHON=$(PYTHON) bash scripts/run_ablation.sh $(ARGS)

# ============ Code quality ============
test:
	PYTHONPATH=$(CURDIR) python -m pytest tests/ -v --tb=short

lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/
	isort --check src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

# ============ Results export ============
export-tables:
	$(EXPORT_RESULTS) --input results/ --output paper/tables/

# ============ Cleanup ============
clean:
	rm -rf outputs/logs/* outputs/checkpoints/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
