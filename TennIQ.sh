#!/bin/bash

show_help() {
    echo "TennIQ"
    echo "==========================================="
    echo "Usage: $0 [--setup|--import|--infer|--demo] [--config FILE|-c FILE] [--input FILE|-i FILE]"
    echo ""
    echo "Commands:"
    echo "  --setup                    Setup virtual environment and dependencies"
    echo "  --import                   Import video from YouTube (only yt is supported as of now)"
    echo "  --infer                    Run inference"
    echo "  --demo                     Run demo inference on existing frames"
    echo ""
    echo "Options:"
    echo "  --config FILE, -c FILE     Configuration file path"
    echo "  --input FILE, -i FILE      Input video file or frames directory"
    echo "  --output FILE, -o FILE     Output video file (for inference)"
    echo "  --no-display               Run without display window"
    echo "  --mode MODE                Analysis mode: 'video' or 'images' (default: video)"
    echo "  --fps FPS                  Presentation FPS for display (default: 30)"
    echo "  --calib-frames NUM         Number of calibration frames"
    echo "  --exp-pred FLOAT           Exponential prediction weight for player tracking"
    echo "  --player-model FILE        Path to custom YOLO model for player tracking"
    echo "  --weights FILE             Path to ball tracking model weights"
    echo "  --model-name NAME          Ball tracking model name"
    echo "  --bounce-model FILE        Path to bounce detection model"
    echo ""
    echo "Examples:"
    echo "  $0 --setup                                    # Setup environment"
    echo "  $0 --import -c config.txt                     # Download video"
    echo "  $0 --infer -c config.txt -i video.mp4        # Analyze video"
    echo "  $0 --demo -c config.txt                       # Demo with frames"
    echo ""
    exit 1
}

setup(){
    echo "Setting up TennIQ..."
    echo "==========================================="

    if [[ ! -f requirements.txt ]]; then
        # Update requirements.txt
        touch requirements.txt
        cat > requirements.txt << EOF
pytube
yt-dlp
numpy
opencv-python
argparse
scipy
matplotlib
scikit-learn
filterpy
ultralytics
catboost
torch
torchvision
EOF
    fi

    if [[ -d tenniq_venv ]]; then
        echo "Virtual environment already exists. Activating..."
    else
        echo "Creating virtual environment..."
        $PYTHON_EXEC -m venv tenniq_venv
    fi

    source tenniq_venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt

    echo ""
    echo "Setup completed successfully!"
    echo "You can now run analysis with: $0 --infer -c <config_file> -i <input>"
}

# Parse arguments
ACTION=""
CONFIG_FILE=""
INPUT_FILE=""
OUTPUT_FILE=""
NO_DISPLAY=""
MODE="video"
FPS=""
CALIB_FRAMES=""
EXP_PRED=""
PLAYER_MODEL=""
WEIGHTS=""
MODEL_NAME=""
BOUNCE_MODEL=""


if [[ -z "$PYTHON_EXEC" ]]; then
    echo "Warning: PYTHON_EXEC environment variable not set. Defaulting to 'python3'."
    PYTHON_EXEC="python3"
fi
$PYTHON_EXEC data/web-scrapping/web-scrapping.py --config "$CONFIG_FILE"

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
        --demo)
            ACTION="demo"
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
        --input|-i)
            if [[ -n "$2" ]]; then
                INPUT_FILE="$2"
                shift 2
            else
                echo "Error: --input requires a file argument."
                show_help
            fi
            ;;
        --output|-o)
            if [[ -n "$2" ]]; then
                OUTPUT_FILE="$2"
                shift 2
            else
                echo "Error: --output requires a file argument."
                show_help
            fi
            ;;
        --no-display)
            NO_DISPLAY="--no-display"
            shift
            ;;
        --mode)
            if [[ -n "$2" ]] && [[ "$2" =~ ^(video|images)$ ]]; then
                MODE="$2"
                shift 2
            else
                echo "Error: --mode must be 'video' or 'images'."
                show_help
            fi
            ;;
        --fps)
            if [[ -n "$2" ]]; then
                FPS="--fps $2"
                shift 2
            else
                echo "Error: --fps requires a numeric argument."
                show_help
            fi
            ;;
        --calib-frames)
            if [[ -n "$2" ]]; then
                CALIB_FRAMES="--calib-frames $2"
                shift 2
            else
                echo "Error: --calib-frames requires a numeric argument."
                show_help
            fi
            ;;
        --exp-pred)
            if [[ -n "$2" ]]; then
                EXP_PRED="--exp-pred $2"
                shift 2
            else
                echo "Error: --exp-pred requires a numeric argument."
                show_help
            fi
            ;;
        --player-model)
            if [[ -n "$2" ]]; then
                PLAYER_MODEL="--player-model \"$2\""
                shift 2
            else
                echo "Error: --player-model requires a file argument."
                show_help
            fi
            ;;
        --weights)
            if [[ -n "$2" ]]; then
                WEIGHTS="--weights \"$2\""
                shift 2
            else
                echo "Error: --weights requires a file argument."
                show_help
            fi
            ;;
        --model-name)
            if [[ -n "$2" ]]; then
                MODEL_NAME="--model-name $2"
                shift 2
            else
                echo "Error: --model-name requires a model name argument."
                show_help
            fi
            ;;
        --bounce-model)
            if [[ -n "$2" ]]; then
                BOUNCE_MODEL="--bounce-model \"$2\""
                shift 2
            else
                echo "Error: --bounce-model requires a file argument."
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

# Validation
if [[ "$ACTION" == "import" ]] && [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: --import requires --config argument."
    show_help
fi

if [[ "$ACTION" == "infer" ]] && [[ -z "$INPUT_FILE" ]]; then
    echo "Error: --infer requires --input argument."
    show_help
fi

echo "TennIQ"
echo "==========================================="
echo "Action: $ACTION"
[[ -n "$CONFIG_FILE" ]] && echo "Config: $CONFIG_FILE"
[[ -n "$INPUT_FILE" ]] && echo "Input: $INPUT_FILE"
[[ -n "$OUTPUT_FILE" ]] && echo "Output: $OUTPUT_FILE"
[[ -n "$MODE" ]] && echo "Mode: $MODE"
[[ -n "$FPS" ]] && echo "FPS: $FPS"
[[ -n "$CALIB_FRAMES" ]] && echo "Calib Frames: $CALIB_FRAMES"
[[ -n "$EXP_PRED" ]] && echo "Exp Pred: $EXP_PRED"
[[ -n "$PLAYER_MODEL" ]] && echo "Player Model: $PLAYER_MODEL"
[[ -n "$WEIGHTS" ]] && echo "Weights: $WEIGHTS"
[[ -n "$MODEL_NAME" ]] && echo "Model Name: $MODEL_NAME"
[[ -n "$BOUNCE_MODEL" ]] && echo "Bounce Model: $BOUNCE_MODEL"
echo ""

if [[ "$ACTION" != "setup" ]]; then
    if [[ ! -d "tenniq_venv" ]]; then
        echo "Virtual environment not found. Please run: $0 --setup"
        exit 1
    fi
    source tenniq_venv/bin/activate
fi

if [[ "$ACTION" == "setup" ]]; then
    setup
elif [[ "$ACTION" == "import" ]]; then
    echo "Importing video from YouTube..."
    $PYTHON_EXEC data/web-scrapping/web-scrapping.py --config "$CONFIG_FILE"
elif [[ "$ACTION" == "infer" ]]; then
    echo "Running tennis analysis inference..."
    INFER_CMD="$PYTHON_EXEC -m inference.inference_main --input \"$INPUT_FILE\" --mode $MODE"
    [[ -n "$CONFIG_FILE" ]] && INFER_CMD="$INFER_CMD --config \"$CONFIG_FILE\""
    [[ -n "$OUTPUT_FILE" ]] && INFER_CMD="$INFER_CMD --output \"$OUTPUT_FILE\""
    [[ -n "$NO_DISPLAY" ]] && INFER_CMD="$INFER_CMD $NO_DISPLAY"
    [[ -n "$FPS" ]] && INFER_CMD="$INFER_CMD $FPS"
    [[ -n "$CALIB_FRAMES" ]] && INFER_CMD="$INFER_CMD $CALIB_FRAMES"
    [[ -n "$EXP_PRED" ]] && INFER_CMD="$INFER_CMD $EXP_PRED"
    [[ -n "$PLAYER_MODEL" ]] && INFER_CMD="$INFER_CMD $PLAYER_MODEL"
    [[ -n "$WEIGHTS" ]] && INFER_CMD="$INFER_CMD $WEIGHTS"
    [[ -n "$MODEL_NAME" ]] && INFER_CMD="$INFER_CMD $MODEL_NAME"
    [[ -n "$BOUNCE_MODEL" ]] && INFER_CMD="$INFER_CMD $BOUNCE_MODEL"

    eval $INFER_CMD
elif [[ "$ACTION" == "demo" ]]; then
    echo "Running demo analysis..."

    # FRAMES_DIR="data/web-scrapping/frames_test"
    FRAMES_DIR="TrackNetv4/data/tennis/Dataset/game9/Clip2"
    if [[ ! -d "$FRAMES_DIR" ]]; then
        echo "Error: Frames directory not found: $FRAMES_DIR"
        echo "Please run: $0 --import -c <config_file> first"
        exit 1
    fi
    
    if [[ -z "$CONFIG_FILE" ]]; then
        CONFIG_FILE="inference/config/config_test.txt"
    fi
    
    DEMO_CMD="$PYTHON_EXEC -m inference.inference_main --config \"$CONFIG_FILE\" --input \"$FRAMES_DIR\" --mode images"
    
    [[ -n "$FPS" ]] && DEMO_CMD="$DEMO_CMD $FPS"
    [[ -n "$CALIB_FRAMES" ]] && DEMO_CMD="$DEMO_CMD $CALIB_FRAMES"
    [[ -n "$EXP_PRED" ]] && DEMO_CMD="$DEMO_CMD $EXP_PRED"
    [[ -n "$NO_DISPLAY" ]] && DEMO_CMD="$DEMO_CMD $NO_DISPLAY"
    if [[ -n "$MODEL_NAME" ]]; then
        DEMO_CMD="$DEMO_CMD $MODEL_NAME"
    else
        echo "Available model options:"
        echo "  --model-name yolo --weights \"yolo_train/ball_detection/yolo12n_ball_fine_tune3/weights/best.pt\""
        echo "  --model-name TrackNetV4_TypeA --weights \"models/TrackNetV4_TypeA_epoch_47.pth\""
        DEMO_CMD="$DEMO_CMD --model-name yolo"
    fi

    if [[ -n "$WEIGHTS" ]]; then
        DEMO_CMD="$DEMO_CMD $WEIGHTS"
    else
        if [[ "$DEMO_CMD" == *"--model-name yolo"* ]]; then
            DEMO_CMD="$DEMO_CMD --weights \"yolo_train/ball_detection/yolo12n_ball_fine_tune3/weights/best.pt\""
        elif [[ "$DEMO_CMD" == *"--model-name TrackNetV4_TypeA"* ]]; then
            DEMO_CMD="$DEMO_CMD --weights \"models/TrackNetV4_TypeA_epoch_47.pth\""
        else
            # Fallback to YOLO
            DEMO_CMD="$DEMO_CMD --weights \"yolo_train/ball_detection/yolo12n_ball_fine_tune3/weights/best.pt\""
        fi
    fi

    if [[ -n "$PLAYER_MODEL" ]]; then
        DEMO_CMD="$DEMO_CMD $PLAYER_MODEL"
    else
        DEMO_CMD="$DEMO_CMD --player-model \"yolo_train/player_detection/yolo11n_player_finetune4/weights/best.pt\""
    fi
    
    if [[ -n "$BOUNCE_MODEL" ]]; then
        DEMO_CMD="$DEMO_CMD $BOUNCE_MODEL"
    else
        DEMO_CMD="$DEMO_CMD --bounce-model \"models/ctb_regr_bounce.cbm\""
    fi
    
    echo "Running: $DEMO_CMD"
    echo "==========================================="
    eval $DEMO_CMD
fi