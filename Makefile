SHELL := /bin/bash

JETSON_HOST ?= nvidia@192.168.1.230
JETSON_DIR ?= /home/nvidia/GPT-SoVITS
MODEL_SOURCE ?= ModelScope
PREPARE_MODELS ?= false

.PHONY: jetson-sync jetson-models jetson-build jetson-up jetson-pipeline

jetson-sync:
	JETSON_HOST="$(JETSON_HOST)" JETSON_DIR="$(JETSON_DIR)" \
		bash scripts/jetson/sync_to_jetson.sh

jetson-models:
	JETSON_HOST="$(JETSON_HOST)" JETSON_DIR="$(JETSON_DIR)" MODEL_SOURCE="$(MODEL_SOURCE)" \
		bash scripts/jetson/prepare_models_on_jetson.sh

jetson-build:
	JETSON_HOST="$(JETSON_HOST)" JETSON_DIR="$(JETSON_DIR)" \
		bash scripts/jetson/build_on_jetson.sh

jetson-up:
	JETSON_HOST="$(JETSON_HOST)" JETSON_DIR="$(JETSON_DIR)" \
		bash scripts/jetson/run_service_on_jetson.sh

jetson-pipeline:
	JETSON_HOST="$(JETSON_HOST)" JETSON_DIR="$(JETSON_DIR)" \
		PREPARE_MODELS="$(PREPARE_MODELS)" MODEL_SOURCE="$(MODEL_SOURCE)" \
		bash scripts/jetson/auto_pipeline_on_jetson.sh
