#!/bin/bash

show_help() {
    echo "Usage: $0 [--import|--infer] [--config FILE|-c FILE]"
    exit 1
}

setup(){
    echo "Setting up the environment..."
    if [[ ! -f requirements.txt ]]; then
        touch requirements.txt
        echo "pytube==15.0.0" >> requirements.txt
        echo "yt-dlp==2025.7.21" >> requirements.txt
        echo "numpy==1.23.5" >> requirements.txt
        echo "argparse" >> requirements.txt
        echo "opencv-python" >> requirements.txt
    fi

    if [[ -d tenniq_venv ]]; then
        echo "Virtual environment already exists. Activating..."
    else
        echo "Creating virtual environment..."
        python3 -m venv tenniq_venv
    fi
    source tenniq_venv/bin/activate
    pip install -r requirements.txt
}

# Parse arguments
ACTION=""
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup)
            ACTION="setup"
            shift
            ;;
        --import)
            ACTION="import"
            shift
            ;;
        --infer)
            ACTION="infer"
            shift
            ;;
        --config|-c)
            if [[ -n "$2" ]]; then
                CONFIG_FILE="$2"
                shift 2
            else
                echo "Error: --config requires a file argument."
                show_help
            fi
            ;;
        --help)
            show_help
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            show_help
            ;;
    esac
done

if [[ -z "$ACTION" ]]; then
    show_help
fi

if [[ "$ACTION" == "import" || "$ACTION" == "infer" ]] && [[ -z "$CONFIG_FILE" ]]; then
    show_help
fi

echo "Action: $ACTION"
echo "Config file: $CONFIG_FILE"

if [[ "$ACTION" == "setup" ]]; then
    setup
elif [[ "$ACTION" == "import" ]]; then
    python3 data/web-scrapping/web-scrapping.py --config "$CONFIG_FILE"

elif [[ "$ACTION" == "infer" ]]; then
    python3 inference/inference_main.py --config "$CONFIG_FILE"
fi