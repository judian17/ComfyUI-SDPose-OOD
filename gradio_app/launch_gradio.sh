#!/bin/bash
# Launch script for SDPose Gradio App
# Author: T. S. Liang, Oct. 2025

echo "🚀 Launching SDPose Gradio App..."
echo "📍 Make sure you're in the correct conda/virtual environment"
echo ""

# Default settings
SHARE=false
SERVER_NAME="0.0.0.0"
SERVER_PORT=7860

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --share)
            SHARE=true
            shift
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --host)
            SERVER_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--share] [--port PORT] [--host HOST]"
            exit 1
            ;;
    esac
done

# Build command
CMD="python SDPose_gradio.py --server_name $SERVER_NAME --server_port $SERVER_PORT"
if [ "$SHARE" = true ]; then
    CMD="$CMD --share"
fi

echo "🌐 Server will run on: http://$SERVER_NAME:$SERVER_PORT"
if [ "$SHARE" = true ]; then
    echo "🔗 Public link will be generated"
fi
echo ""

# Run the app
$CMD

