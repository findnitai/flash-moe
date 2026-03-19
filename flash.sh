#!/bin/bash

# Default values
MODE=${2:-speed} # Default to speed
K=4
BIT="--2bit"

if [ "$MODE" == "accuracy" ]; then
    K=8
    BIT=""
fi

if [ "$1" == "server" ]; then
    echo "------------------------------------------------"
    echo "Starting Server in $MODE mode (K=$K)..."
    echo "------------------------------------------------"
    ./metal_infer/infer --serve 8000 \
                        --weights qwen-122b-4bit/model_weights.bin \
                        --manifest qwen-122b-4bit/model_weights.json \
                        --vocab qwen-122b-4bit/vocab.bin \
                        --model qwen-122b-4bit \
                        $BIT --k $K --think-budget 250
elif [ "$1" == "chat" ]; then
    echo "------------------------------------------------"
    echo "Starting Chat Client in $MODE mode (K=$K)..."
    echo "------------------------------------------------"
    ./metal_infer/chat --k $K
else
    echo "Usage: ./flash.sh [server|chat] [speed|accuracy]"
    echo ""
    echo "Example: ./flash.sh server speed      # Fast 2-bit mode"
    echo "Example: ./flash.sh chat accuracy     # Stats for K=8 mode"
fi
