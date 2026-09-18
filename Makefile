.PHONY: setup-mac setup-cuda docker-build train train-baseline train-debug \
        train-100 train-2000 \
        evaluate benchmark benchmark-baseline \
        ablation-bottleneck ablation-depth ablation-loss ablation-all \
        test lint format export-tables clean \
		train-dinov3-2000 train-dinov3-2000-dino-loss compare-dinov3-loss

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
	python scripts/train.py $(ARGS)

train-baseline:
	python scripts/train.py experiment_name=dino_cnn_cnn $(ARGS)

train-debug:
	python scripts/train.py experiment_name=smoke_dino_cnn_cnn \
		training.max_epochs=2 data.batch_size=2 \
		data.train_subset_size=8 data.test_subset_size=4 $(ARGS)

train-100:
	python scripts/train.py experiment_name=dino_cnn_cnn_100 \
		data.train_subset_size=100 $(ARGS)

train-2000:
	python scripts/train.py experiment_name=dino_cnn_cnn_2000 \
		data.train_subset_size=2000 $(ARGS)

train-dinov3-2000:
	python scripts/train.py ablation=a_loss_l1_only \
		experiment_name=dinov3_2000img encoder=dinov3_b \
		data.train_subset_size=2000 training.max_epochs=100 \
		training.early_stopping.patience=15 $(ARGS)

train-dinov3-2000-dino-loss:
	python scripts/train.py ablation=a_loss_l1_dino_feature \
		experiment_name=dinov3_2000img_dino_loss encoder=dinov3_b \
		data.train_subset_size=2000 training.max_epochs=100 \
		training.early_stopping.patience=15 $(ARGS)

compare-dinov3-loss:
	python scripts/train.py ablation=a_loss_l1_only \
		experiment_name=dinov3_2000img encoder=dinov3_b \
		data.train_subset_size=2000 training.max_epochs=100 \
		training.early_stopping.patience=15 $(ARGS)
	python scripts/train.py ablation=a_loss_l1_dino_feature \
		experiment_name=dinov3_2000img_dino_loss encoder=dinov3_b \
		data.train_subset_size=2000 training.max_epochs=100 \
		training.early_stopping.patience=15 $(ARGS)

docker-train:
	docker run --gpus all \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/outputs:/workspace/outputs \
		rwkv-color $(ARGS)

# ============ Evaluation ============
evaluate:
	python scripts/evaluate.py $(ARGS)

benchmark:
	python scripts/benchmark_efficiency.py $(ARGS)

benchmark-baseline:
	python scripts/benchmark_efficiency.py --bottleneck cnn \
		--encoder-dim 768 --patch-size 14 $(ARGS)

# ============ Ablation batch runs ============
ablation-bottleneck:
	@for bn in none cnn; do \
		echo "=== Running bottleneck=$$bn ===" ; \
		python scripts/train.py encoder=dinov2_b decoder=cnn bottleneck=$$bn \
			experiment_name=A_bn_$$bn $(ARGS) ; \
	done

ablation-depth:
	@for d in 2 4 6; do \
		echo "=== Running CNN depth=$$d ===" ; \
		python scripts/train.py encoder=dinov2_b decoder=cnn bottleneck=cnn \
			bottleneck.num_layers=$$d experiment_name=A_cnn_depth_$$d $(ARGS) ; \
	done

ablation-loss:
	python scripts/train.py experiment_name=A_loss_stage1 $(ARGS)
	python scripts/train.py ablation=a_loss_l1_only experiment_name=A_loss_l1 $(ARGS)
	python scripts/train.py ablation=a_loss_l1_ssim experiment_name=A_loss_l1_ssim $(ARGS)

ablation-all:
	bash scripts/run_ablation.sh $(ARGS)

# ============ Code quality ============
test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/
	isort --check src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

# ============ Results export ============
export-tables:
	python scripts/export_results.py --input results/ --output paper/tables/

# ============ Cleanup ============
clean:
	rm -rf outputs/logs/* outputs/checkpoints/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
