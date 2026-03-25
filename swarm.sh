#!/usr/bin/env bash
#
# swarm.sh - CLI tool for running the Swarm LLM system
#
# Usage:
#   ./swarm.sh build              Build the Docker image
#   ./swarm.sh run [options]      Run the swarm (see options below)
#   ./swarm.sh resume <file>      Resume a swarm from a JSON state file
#   ./swarm.sh logs               Show recent log files
#   ./swarm.sh clean              Clean up output files
#   ./swarm.sh shell              Open a shell in the container
#   ./swarm.sh help               Show this help message
#
# Run Options:
#   -c, --cycles N             Maximum cycles to run (default: 50)
#   -a, --agents SPECS         Agent specs as MODEL:PROMPT or just PROMPT
#   -r, --resume FILE          Resume from a previous run's JSON file
#   -m, --model MODEL          Default LLM model to use
#   -b, --char-budget N        Character budget per agent turn (default: from .env)
#   -w, --watch                Show real-time agent activity (tool calls, messages)
#   -v, --verbose              Enable verbose logging
#   -q, --quiet                Only show cycle summaries
#
# Agent Specs:
#   Agents can be specified as MODEL:PROMPT or just PROMPT.
#   - PROMPT alone uses the default model (LLM_MODEL env var)
#   - MODEL:PROMPT uses a specific model for that agent
#   Model nicknames (claude, gemini, etc.) are resolved via MODEL_* env vars.
#
# Examples:
#   ./swarm.sh run                                    # Run with defaults
#   ./swarm.sh run -c 20                              # Run for 20 cycles
#   ./swarm.sh run -a curious_explorer observer       # Specific agents
#   ./swarm.sh run -a claude:minimal gemini:minimal   # Different models per agent
#   ./swarm.sh resume output/swarm_20250116.json      # Resume from state file
#   ./swarm.sh run -r output/swarm_20250116.json -c 10  # Resume with more cycles
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="swarm-llm"
CONTAINER_NAME="swarm-llm-$(date +%s)"
OUTPUT_DIR="./output"
ENV_FILE=".env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════╗"
    echo "║     SWARM LLM - Emergent Multi-Agent System       ║"
    echo "╚═══════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_help() {
    print_header
    echo "Usage: ./swarm.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build              Build the Docker image"
    echo "  run [options]      Run the swarm"
    echo "  resume <file>      Resume from a previous run's JSON state file"
    echo "  logs               Show recent log files"
    echo "  clean              Clean up output files"
    echo "  shell              Open a shell in the container"
    echo "  help               Show this help message"
    echo ""
    echo "Run Options:"
    echo "  -c, --cycles N             Maximum cycles (default: from .env or 50)"
    echo "  -a, --agents SPECS...      Agent specs to spawn (see Agent Specs below)"
    echo "  -r, --resume FILE          Resume from a previous run's JSON file"
    echo "  -m, --model MODEL          Default LLM model to use"
    echo "  -b, --char-budget N        Character budget per agent turn (default: from .env)"
    echo "  -o, --output PREFIX        Output file prefix"
    echo "  -w, --watch                Show real-time agent activity"
    echo "  -v, --verbose              Enable verbose logging"
    echo "  -q, --quiet                Only show cycle summaries"
    echo ""
    echo "Agent Specs:"
    echo "  Agents are specified as MODEL:PROMPT or just PROMPT."
    echo "  - PROMPT alone uses the default model (from LLM_MODEL env var)"
    echo "  - MODEL:PROMPT uses a specific model for that agent"
    echo ""
    echo "  Model nicknames (claude, gemini, opus, etc.) are resolved via"
    echo "  MODEL_* env vars in .env (e.g., MODEL_CLAUDE=claude-sonnet-4-5-20250929)"
    echo ""
    echo "Examples:"
    echo "  ./swarm.sh build"
    echo "  ./swarm.sh run"
    echo "  ./swarm.sh run -a claude:minimal gemini:minimal    # Mixed models"
    echo "  ./swarm.sh run -a opus:coordinator claude:observer # Per-agent models"
    echo "  ./swarm.sh resume output/swarm_20250116_120000.json"
    echo "  ./swarm.sh run -r output/swarm_20250116.json -c 10 # Resume + more cycles"
    echo ""
    echo "Environment:"
    echo "  Copy .env.example to .env and configure your API keys."
    echo "  Output files are written to ./output/"
}

check_env() {
    if [ ! -f "$ENV_FILE" ]; then
        echo -e "${YELLOW}Warning: .env file not found.${NC}"
        echo "Creating from .env.example..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            echo -e "${YELLOW}Please edit .env and add your API keys.${NC}"
            exit 1
        else
            echo -e "${RED}Error: .env.example not found.${NC}"
            exit 1
        fi
    fi

    # Check for API keys
    if ! grep -q "^ANTHROPIC_API_KEY=.*[^=]$" "$ENV_FILE" && \
       ! grep -q "^OPENAI_API_KEY=.*[^=]$" "$ENV_FILE" && \
       ! grep -q "^GOOGLE_API_KEY=.*[^=]$" "$ENV_FILE"; then
        echo -e "${YELLOW}Warning: No API keys configured in .env${NC}"
        echo "Please add at least one of:"
        echo "  - ANTHROPIC_API_KEY"
        echo "  - OPENAI_API_KEY"
        echo "  - GOOGLE_API_KEY"
    fi
}

build_image() {
    print_header
    echo -e "${GREEN}Building Docker image...${NC}"
    docker build -t "$IMAGE_NAME:latest" .
    echo -e "${GREEN}Build complete!${NC}"
}

run_swarm() {
    print_header
    check_env

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    echo -e "${GREEN}Starting swarm...${NC}"
    echo "Output directory: $OUTPUT_DIR"
    echo ""

    # Run the container
    docker run --rm -it \
        --name "$CONTAINER_NAME" \
        --env-file "$ENV_FILE" \
        -v "$(pwd)/$OUTPUT_DIR:/app/output" \
        "$IMAGE_NAME:latest" \
        "$@"
}

show_logs() {
    print_header
    echo -e "${GREEN}Recent log files:${NC}"
    echo ""

    if [ -d "$OUTPUT_DIR" ]; then
        # Find recent log files
        LOG_FILES=$(find "$OUTPUT_DIR" -name "*.log" -type f -mtime -1 | sort -r | head -5)

        if [ -z "$LOG_FILES" ]; then
            echo "No recent log files found."
        else
            for f in $LOG_FILES; do
                echo -e "${BLUE}=== $f ===${NC}"
                tail -20 "$f"
                echo ""
            done
        fi

        # Show summary files
        SUMMARY_FILES=$(find "$OUTPUT_DIR" -name "*_summary.txt" -type f -mtime -1 | sort -r | head -3)
        if [ -n "$SUMMARY_FILES" ]; then
            echo -e "${GREEN}Recent summaries:${NC}"
            for f in $SUMMARY_FILES; do
                echo -e "${BLUE}=== $f ===${NC}"
                cat "$f"
                echo ""
            done
        fi
    else
        echo "Output directory not found."
    fi
}

clean_output() {
    print_header
    echo -e "${YELLOW}Cleaning output directory...${NC}"

    if [ -d "$OUTPUT_DIR" ]; then
        read -p "Delete all files in $OUTPUT_DIR? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "${OUTPUT_DIR:?}"/*
            echo -e "${GREEN}Cleaned!${NC}"
        else
            echo "Cancelled."
        fi
    else
        echo "Output directory not found."
    fi
}

open_shell() {
    print_header
    check_env

    echo -e "${GREEN}Opening shell in container...${NC}"
    docker run --rm -it \
        --name "${CONTAINER_NAME}-shell" \
        --env-file "$ENV_FILE" \
        -v "$(pwd)/$OUTPUT_DIR:/app/output" \
        --entrypoint /bin/bash \
        "$IMAGE_NAME:latest"
}

resume_swarm() {
    print_header
    check_env

    if [ -z "$1" ]; then
        echo -e "${RED}Error: No state file specified.${NC}"
        echo "Usage: ./swarm.sh resume <json_file> [options]"
        echo ""
        echo "Example: ./swarm.sh resume output/swarm_20250116_120000.json"
        exit 1
    fi

    STATE_FILE="$1"
    shift

    # Check if file exists (handle both absolute and relative paths)
    if [ ! -f "$STATE_FILE" ]; then
        # Try in output directory
        if [ -f "$OUTPUT_DIR/$STATE_FILE" ]; then
            STATE_FILE="$OUTPUT_DIR/$STATE_FILE"
        else
            echo -e "${RED}Error: State file not found: $STATE_FILE${NC}"
            exit 1
        fi
    fi

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    echo -e "${GREEN}Resuming swarm from: $STATE_FILE${NC}"
    echo "Output directory: $OUTPUT_DIR"
    echo ""

    # Convert to path relative to /app for container
    # If the file is in output/, use /app/output/filename
    CONTAINER_PATH="$STATE_FILE"
    if [[ "$STATE_FILE" == "$OUTPUT_DIR/"* ]] || [[ "$STATE_FILE" == "./output/"* ]]; then
        BASENAME=$(basename "$STATE_FILE")
        CONTAINER_PATH="/app/output/$BASENAME"
    elif [[ "$STATE_FILE" == "output/"* ]]; then
        BASENAME=$(basename "$STATE_FILE")
        CONTAINER_PATH="/app/output/$BASENAME"
    fi

    # Run the container with --resume flag
    docker run --rm -it \
        --name "$CONTAINER_NAME" \
        --env-file "$ENV_FILE" \
        -v "$(pwd)/$OUTPUT_DIR:/app/output" \
        "$IMAGE_NAME:latest" \
        --resume "$CONTAINER_PATH" "$@"
}

# Main command handler
case "${1:-help}" in
    build)
        build_image
        ;;
    run)
        shift
        run_swarm "$@"
        ;;
    resume)
        shift
        resume_swarm "$@"
        ;;
    logs)
        show_logs
        ;;
    clean)
        clean_output
        ;;
    shell)
        open_shell
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Run './swarm.sh help' for usage."
        exit 1
        ;;
esac
