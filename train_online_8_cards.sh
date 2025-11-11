#!/bin/bash
exec > >(tee train_a6000.log) 2>&1
# PasoDoble Training Script
# This script runs the proposer-solver framework training

set -e  # Exit on any error
set -u  # Exit on undefined variables
set -o pipefail  # Exit on pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables for process tracking
PROPOSER_VLLM_PID=""
SOLVER_VLLM_PID=""

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to kill process by PID with timeout
kill_process_safe() {
    local pid=$1
    local name=$2
    local timeout=${3:-10}
    
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log_info "Terminating $name (PID: $pid)..."
        
        # Try graceful termination first
        kill -TERM "$pid" 2>/dev/null || true
        
        # Wait for process to terminate
        local count=0
        while [ $count -lt $timeout ] && kill -0 "$pid" 2>/dev/null; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            log_warn "$name did not terminate gracefully, force killing..."
            kill -KILL "$pid" 2>/dev/null || true
            sleep 2
        fi
        
        if kill -0 "$pid" 2>/dev/null; then
            log_error "Failed to terminate $name (PID: $pid)"
        else
            log_success "$name terminated successfully"
        fi
    fi
}

# Function to check if port is in use
check_port() {
    local host=$1
    local port=$2
    
    # Try multiple detection methods
    if command -v netstat &> /dev/null; then
        # Check for port binding (including 0.0.0.0)
        netstat -tuln | grep -E ":$port\s" >/dev/null 2>&1
    elif command -v ss &> /dev/null; then
        # Check for port binding (including 0.0.0.0)  
        ss -tuln | grep -E ":$port\s" >/dev/null 2>&1
    else
        # Fallback: try to connect to the port
        timeout 5 bash -c "</dev/tcp/$host/$port" >/dev/null 2>&1
    fi
}

# Function to check if VLLM server is ready (HTTP health check)
check_vllm_health() {
    local host=$1
    local port=$2
    
    # Try to get server info via HTTP
    if command -v curl &> /dev/null; then
        curl -s --connect-timeout 5 --max-time 10 "http://$host:$port/v1/models" >/dev/null 2>&1
    elif command -v wget &> /dev/null; then
        wget -q --timeout=10 --tries=1 -O /dev/null "http://$host:$port/v1/models" >/dev/null 2>&1
    else
        # Fallback to simple TCP connection
        timeout 5 bash -c "</dev/tcp/$host/$port" >/dev/null 2>&1
    fi
}

# Function to wait for VLLM server to be ready
wait_for_vllm_server() {
    local host=$1
    local port=$2
    local timeout=${3:-180}  # Increased default timeout
    local count=0
    
    log_info "Waiting for VLLM server at $host:$port to become ready..."
    
    while [ $count -lt $timeout ]; do
        # First check if port is listening
        if check_port "$host" "$port"; then
            log_info "Port $port is listening, checking server health..."
            
            # Then check if VLLM server is actually ready
            if check_vllm_health "$host" "$port"; then
                log_success "VLLM server at $host:$port is ready!"
                return 0
            else
                log_info "Port is open but server not ready yet, continuing to wait..."
            fi
        else
            log_info "Port $port not yet listening (waited ${count}s)..."
        fi
        
        sleep 5
        count=$((count + 5))
    done
    
    log_error "Timeout waiting for VLLM server at $host:$port (waited ${timeout}s)"
    
    # Provide debugging information
    log_info "Debug information:"
    if command -v netstat &> /dev/null; then
        log_info "Ports currently in use:"
        netstat -tuln | grep -E ":(8080|8081)\s" || log_info "No ports 8080/8081 found"
    fi
    
    if command -v ps &> /dev/null; then
        log_info "VLLM processes:"
        ps aux | grep "vllm-serve" | grep -v grep || log_info "No VLLM processes found"
    fi
    
    return 1
}

# Function to force release port
force_release_port() {
    local port=$1
    
    log_info "Force releasing port $port..."
    
    # Find and kill processes using the port
    if command -v lsof &> /dev/null; then
        local pids=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pids" ]; then
            log_info "Found processes using port $port: $pids"
            echo "$pids" | xargs -r kill -TERM 2>/dev/null || true
            sleep 3
            echo "$pids" | xargs -r kill -KILL 2>/dev/null || true
        fi
    elif command -v fuser &> /dev/null; then
        fuser -k $port/tcp 2>/dev/null || true
    fi
}

# Enhanced cleanup function
cleanup() {
    log_info "Starting cleanup process..."
    
    # Kill VLLM servers
    if [ -n "$PROPOSER_VLLM_PID" ]; then
        kill_process_safe "$PROPOSER_VLLM_PID" "Proposer VLLM Server"
    fi
    
    if [ -n "$SOLVER_VLLM_PID" ]; then
        kill_process_safe "$SOLVER_VLLM_PID" "Solver VLLM Server"
    fi
    
    # Force release ports if they're still occupied
    if check_port "$PROPOSER_VLLM_SERVER_HOST" "$PROPOSER_VLLM_SERVER_PORT"; then
        force_release_port "$PROPOSER_VLLM_SERVER_PORT"
    fi
    
    if check_port "$PROPOSER_VLLM_SERVER_HOST" "$PROPOSER_VLLM_CLIENT_PORT"; then
        force_release_port "$PROPOSER_VLLM_CLIENT_PORT"
    fi

    if check_port "$SOLVER_VLLM_SERVER_HOST" "$SOLVER_VLLM_SERVER_PORT"; then
        force_release_port "$SOLVER_VLLM_SERVER_PORT"
    fi

    if check_port "$SOLVER_VLLM_SERVER_HOST" "$SOLVER_VLLM_CLIENT_PORT"; then
        force_release_port "$SOLVER_VLLM_CLIENT_PORT"
    fi
    
    # Additional cleanup for any remaining trl vllm-serve processes
    log_info "Cleaning up any remaining VLLM processes..."
    pkill -f "trl vllm-serve" 2>/dev/null || true
    
    log_success "Cleanup completed"
    exit 0
}

# Set trap for cleanup on script exit
trap cleanup EXIT INT TERM

# Environment setup
export DS_SKIP_CUDA_CHECK=1
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
# export PATH="/usr/local/cuda-12.1/bin:$PATH"
# : "${LD_LIBRARY_PATH:=}"
# export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
# export CUDA_HOME="/usr/local/cuda-12.1"

export WANDB_DATA_DIR="wandb"
export TRITON_CACHE_DIR=triton
export TORCHINDUCTOR_CACHE_DIR=ti-cache
export TMPDIR=tmp
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="PasoDoble-training"

# Default configuration
PROPOSER_MODEL_NAME="your_proposer_sft_checkpoint_name_or_path"

SOLVER_MODEL_NAME="your_solver_sft_checkpoint_name_or_path"

PROPOSER_DEEPSPEED_CONFIG="./configs/proposer_deepspeed_config_6_cards.json"
SOLVER_DEEPSPEED_CONFIG="./configs/solver_deepspeed_config_6_cards.json"
WANDB_RUN_NAME="PasoDoble-$(date +%Y%m%d-%H%M%S)_online"
OUTPUT_DIR_PROPOSER="output/proposer_$(date +%Y%m%d-%H%M%S)_online"
OUTPUT_DIR_SOLVER="output/solver_$(date +%Y%m%d-%H%M%S)_online"
MAX_STEPS=2000
SAVE_STEPS=40
PROPOSER_NUM_GENERATIONS=6
SOLVER_NUM_GENERATIONS=6
USE_KNOWLEDGE=1
USE_DS=1
LOSS_TYPE="dr_grpo"
BETA=0.0

# VLLM Configuration
USE_VLLM=1
VLLM_MODE="server"
PROPOSER_VLLM_SERVER_HOST="127.0.0.5"
PROPOSER_VLLM_SERVER_PORT="8106"
PROPOSER_VLLM_CLIENT_PORT="51218"
SOLVER_VLLM_SERVER_HOST="127.0.0.6"
SOLVER_VLLM_SERVER_PORT="8107"
SOLVER_VLLM_CLIENT_PORT="51219"
VLLM_TENSOR_PARALLEL_SIZE=6
VLLM_GPU_MEMORY_UTILIZATION=0.95

# Model Configuration
MAX_PROPOSER_PROMPT_LENGTH=1280
MAX_SOLVER_PROMPT_LENGTH=512
MAX_PROPOSER_COMPLETION_LENGTH=6144
MAX_SOLVER_COMPLETION_LENGTH=6144

# Training Configuration
PROPOSER_TEMPERATURE=0.6
SOLVER_TEMPERATURE=0.6
PASSING_RATE_LOWER_THRESHOLD=0.2
PASSING_RATE_UPPER_THRESHOLD=1.0

# Function to show usage
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --proposer_model_name MODEL               Pretrained model name or path (default: $PROPOSER_MODEL_NAME)"
    echo "  --solver_model_name MODEL               Pretrained model name or path (default: $SOLVER_MODEL_NAME)"
    echo "  --proposer_deepspeed_path PATH   Path to proposer deepspeed config file"
    echo "  --solver_deepspeed_path PATH     Path to solver deepspeed config file"
    echo "  --wandb_run_name NAME            Wandb run name"
    echo "  --output_dir_proposer DIR        Output directory for proposer"
    echo "  --output_dir_solver DIR          Output directory for solver"
    echo "  --max_steps STEPS                Maximum training steps (default: $MAX_STEPS)"
    echo "  --save_steps STEPS               Steps between checkpoints (default: $SAVE_STEPS)"
    echo "  --proposer_temp TEMP             Proposer sampling temperature (default: $PROPOSER_TEMPERATURE)"
    echo "  --solver_temp TEMP               Solver sampling temperature (default: $SOLVER_TEMPERATURE)"
    echo "  --passing_rate_lower_threshold RATE                 Passing rate lower threshold (default: $PASSING_RATE_LOWER_THRESHOLD)"
    echo "  --passing_rate_upper_threshold RATE                 Passing rate upper threshold (default: $PASSING_RATE_UPPER_THRESHOLD)"
    echo "  --proposer_vllm_host HOST                VLLM server host (default: $PROPOSER_VLLM_SERVER_HOST)"
    echo "  --proposer_vllm_port PORT                 VLLM server port (default: $PROPOSER_VLLM_SERVER_PORT)"
    echo "  --proposer_vllm_port PORT                 VLLM server port (default: $PROPOSER_VLLM_CLIENT_PORT)"
    echo "  --solver_vllm_host HOST                 VLLM server host (default: $SOLVER_VLLM_SERVER_HOST)"
    echo "  --solver_vllm_port PORT                 VLLM server port (default: $SOLVER_VLLM_SERVER_PORT)"
    echo "  --solver_vllm_port PORT                 VLLM server port (default: $SOLVER_VLLM_CLIENT_PORT)"
    echo "  --gpu_memory UTIL                GPU memory utilization (default: $VLLM_GPU_MEMORY_UTILIZATION)"
    echo "  --use_knowledge FLAG             Use knowledge or not, 0/1 (default: $USE_KNOWLEDGE)"
    echo "  --use_ds FLAG                    Use DeepSpeed or not, 0/1 (default: $USE_DS)"
    echo "  --use_vllm FLAG                  Use VLLM or not, 0/1 (default: $USE_VLLM)"
    echo "  --loss_type TYPE                 Loss type (default: $LOSS_TYPE)"
    echo "  --beta BETA                     Beta for DSR (default: $BETA)"
    echo "  -h, --help                       Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --model_name Qwen/Qwen2.5-7B-Instruct --max_steps 5000 --use_vllm 1"
}

# Function to create default DeepSpeed config
create_deepspeed_config() {
    local config_path="$1"
    local config_name="$2"
    
    log_warn "DeepSpeed config file not found: $config_path"
    log_info "Creating a basic DeepSpeed config for $config_name..."
    
    mkdir -p "$(dirname "$config_path")"
    cat > "$config_path" << 'EOF'
{
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "wall_clock_breakdown": false
}
EOF
    log_success "Created basic DeepSpeed config at: $config_path"
}

# Function to check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check if accelerate is installed
    if ! command -v accelerate &> /dev/null; then
        log_error "accelerate is not installed or not in PATH"
        return 1
    fi
    
    # Check if trl is installed
    if ! command -v trl &> /dev/null; then
        log_error "trl is not installed or not in PATH"
        return 1
    fi
    
    # Check if CUDA is available
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "nvidia-smi not found. GPU support may not be available."
    fi
    
    # Check if train_online_8_cards.py exists
    if [ ! -f "train_online.py" ]; then
        log_error "train_online.py not found in current directory"
        return 1
    fi
    
    # Check if accelerate config exists
    if [ ! -f "configs/accelerate_config_6_cards.yaml" ]; then
        log_warn "configs/accelerate_config_6_cards.yaml not found. Using default accelerate configuration."
    fi
    
    log_success "Dependency check completed"
}

# Function to validate configuration
validate_config() {
    log_info "Validating configuration..."
    
    # Validate numeric parameters
    if ! [[ "$MAX_STEPS" =~ ^[0-9]+$ ]] || [ "$MAX_STEPS" -le 0 ]; then
        log_error "Invalid max_steps: $MAX_STEPS (must be positive integer)"
        return 1
    fi
    
    if ! [[ "$SAVE_STEPS" =~ ^[0-9]+$ ]] || [ "$SAVE_STEPS" -le 0 ]; then
        log_error "Invalid save_steps: $SAVE_STEPS (must be positive integer)"
        return 1
    fi
    
    # Validate temperature values
    if ! [[ "$PROPOSER_TEMPERATURE" =~ ^[0-9]*\.?[0-9]+$ ]] || (( $(echo "$PROPOSER_TEMPERATURE <= 0" | bc -l) )); then
        log_error "Invalid proposer temperature: $PROPOSER_TEMPERATURE (must be positive number)"
        return 1
    fi
    
    if ! [[ "$SOLVER_TEMPERATURE" =~ ^[0-9]*\.?[0-9]+$ ]] || (( $(echo "$SOLVER_TEMPERATURE <= 0" | bc -l) )); then
        log_error "Invalid solver temperature: $SOLVER_TEMPERATURE (must be positive number)"
        return 1
    fi
    
    # Validate threshold
    if ! [[ "$PASSING_RATE_LOWER_THRESHOLD" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        log_error "Invalid passing rate lower threshold: $PASSING_RATE_LOWER_THRESHOLD (must be number)"
        return 1
    fi
    
    if ! [[ "$PASSING_RATE_UPPER_THRESHOLD" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        log_error "Invalid passing rate upper threshold: $PASSING_RATE_UPPER_THRESHOLD (must be number)"
        return 1
    fi
    
        # Validate boolean flags (must be 0 or 1)
    if ! [[ "$USE_KNOWLEDGE" =~ ^[01]$ ]]; then
        log_error "Invalid use_knowledge flag: $USE_KNOWLEDGE (must be 0 or 1)"
        return 1
    fi
    
    if ! [[ "$USE_DS" =~ ^[01]$ ]]; then
        log_error "Invalid use_ds flag: $USE_DS (must be 0 or 1)"
        return 1
    fi
    
    if ! [[ "$USE_VLLM" =~ ^[01]$ ]]; then
        log_error "Invalid use_vllm flag: $USE_VLLM (must be 0 or 1)"
        return 1
    fi
    
    log_success "Configuration validation completed"
}

# Function to start VLLM servers
start_vllm_servers() {
    if [ "$USE_VLLM" -eq 1 ]; then
        log_info "Starting VLLM servers..."
        
        # Create history_record directory if it doesn't exist
        mkdir -p history_record
        
        # Check if ports are already in use
        if check_port "$PROPOSER_VLLM_SERVER_HOST" "$PROPOSER_VLLM_SERVER_PORT"; then
            log_warn "Port $PROPOSER_VLLM_SERVER_PORT is already in use, trying to release it..."
            force_release_port "$PROPOSER_VLLM_SERVER_PORT"
            sleep 3
        fi
        
        if check_port "$PROPOSER_VLLM_SERVER_HOST" "$PROPOSER_VLLM_CLIENT_PORT"; then
            log_warn "Port $PROPOSER_VLLM_CLIENT_PORT is already in use, trying to release it..."
            force_release_port "$PROPOSER_VLLM_CLIENT_PORT"
            sleep 3
        fi

        if check_port "$SOLVER_VLLM_SERVER_HOST" "$SOLVER_VLLM_SERVER_PORT"; then
            log_warn "Port $SOLVER_VLLM_SERVER_PORT is already in use, trying to release it..."
            force_release_port "$SOLVER_VLLM_SERVER_PORT"
            sleep 3
        fi
        
        if check_port "$SOLVER_VLLM_SERVER_HOST" "$SOLVER_VLLM_CLIENT_PORT"; then
            log_warn "Port $SOLVER_VLLM_CLIENT_PORT is already in use, trying to release it..."
            force_release_port "$SOLVER_VLLM_CLIENT_PORT"
            sleep 3
        fi
        
        # Start proposer VLLM server
        log_info "Starting Proposer VLLM server on $PROPOSER_VLLM_SERVER_HOST:$PROPOSER_VLLM_SERVER_PORT..."
        CUDA_VISIBLE_DEVICES="0" trl vllm-serve \
            --model "$PROPOSER_MODEL_NAME" \
            --tensor-parallel-size 1 \
            --data-parallel-size 1 \
            --host "$PROPOSER_VLLM_SERVER_HOST" \
            --port "$PROPOSER_VLLM_SERVER_PORT" \
            --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
            &> history_record/vllm_proposer.out &
        PROPOSER_VLLM_PID=$!

        # Start solver VLLM server
        log_info "Starting Solver VLLM server on $SOLVER_VLLM_SERVER_HOST:$SOLVER_VLLM_SERVER_PORT..."
        CUDA_VISIBLE_DEVICES="1" trl vllm-serve \
            --model "$SOLVER_MODEL_NAME" \
            --tensor-parallel-size 1 \
            --data-parallel-size 1 \
            --host "$SOLVER_VLLM_SERVER_HOST" \
            --port "$SOLVER_VLLM_SERVER_PORT" \
            --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
            &> history_record/vllm_solver.out &
        SOLVER_VLLM_PID=$!
        
        log_info "VLLM servers started with PIDs: Proposer=$PROPOSER_VLLM_PID, Solver=$SOLVER_VLLM_PID"
        log_info "Server logs available at:"
        log_info "  Proposer: history_record/vllm_proposer.out"
        log_info "  Solver: history_record/vllm_solver.out"
        
        # Wait for servers to be ready (increased timeout and better checking)
        log_info "Waiting for VLLM servers to initialize (this may take several minutes)..."
        
        if ! wait_for_vllm_server "$PROPOSER_VLLM_SERVER_HOST" "$PROPOSER_VLLM_SERVER_PORT" 900; then
            log_error "Proposer VLLM server failed to become ready"
            log_info "Check the log file for details: history_record/vllm_proposer.out"
            log_info "Last few lines of proposer log:"
            tail -10 history_record/vllm_proposer.out 2>/dev/null || log_info "Could not read log file"
            return 1
        fi
        
        if ! wait_for_vllm_server "$SOLVER_VLLM_SERVER_HOST" "$SOLVER_VLLM_SERVER_PORT" 300; then
            log_error "Solver VLLM server failed to become ready"
            log_info "Check the log file for details: history_record/vllm_solver.out"
            log_info "Last few lines of solver log:"
            tail -10 history_record/vllm_solver.out 2>/dev/null || log_info "Could not read log file"
            return 1
        fi
        
        log_success "Both VLLM servers are ready and responding!"
    else
        log_info "VLLM is disabled, skipping server startup"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --proposer_model_name)
            PROPOSER_MODEL_NAME="$2"
            shift 2
            ;;
        --solver_model_name)
            SOLVER_MODEL_NAME="$2"
            shift 2
            ;;
        --proposer_deepspeed_path)
            PROPOSER_DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --solver_deepspeed_path)
            SOLVER_DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --wandb_run_name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --output_dir_proposer)
            OUTPUT_DIR_PROPOSER="$2"
            shift 2
            ;;
        --output_dir_solver)
            OUTPUT_DIR_SOLVER="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --save_steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --proposer_temp)
            PROPOSER_TEMPERATURE="$2"
            shift 2
            ;;
        --solver_temp)
            SOLVER_TEMPERATURE="$2"
            shift 2
            ;;
        --passing_rate_lower_threshold)
            PASSING_RATE_LOWER_THRESHOLD="$2"
            shift 2
            ;;
        --passing_rate_upper_threshold)
            PASSING_RATE_UPPER_THRESHOLD="$2"
            shift 2
            ;;
        --proposer_vllm_host)
            PROPOSER_VLLM_SERVER_HOST="$2"
            shift 2
            ;;
        --proposer_vllm_port)
            PROPOSER_VLLM_SERVER_PORT="$2"
            shift 2
            ;;
        --proposer_vllm_client_port)
            PROPOSER_VLLM_CLIENT_PORT="$2"
            shift 2
            ;;
        --solver_vllm_host)
            SOLVER_VLLM_SERVER_HOST="$2"
            shift 2
            ;;
        --solver_vllm_port)
            SOLVER_VLLM_SERVER_PORT="$2"
            shift 2
            ;;
        --solver_vllm_client_port)
            SOLVER_VLLM_CLIENT_PORT="$2"
            shift 2
            ;;
        --gpu_memory)
            VLLM_GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --use_knowledge)
            USE_KNOWLEDGE="$2"
            shift 2
            ;;
        --use_ds)
            USE_DS="$2"
            shift 2
            ;;
        --use_vllm)
            USE_VLLM="$2"
            shift 2
            ;;
        --loss_type)
            LOSS_TYPE="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_info "Starting PasoDoble training script..."
    
    # Check dependencies
    if ! check_dependencies; then
        log_error "Dependency check failed"
        exit 1
    fi
    
    # Validate configuration
    if ! validate_config; then
        log_error "Configuration validation failed"
        exit 1
    fi
    
    # Create output directories
    log_info "Creating output directories..."
    mkdir -p "$OUTPUT_DIR_PROPOSER"
    mkdir -p "$OUTPUT_DIR_SOLVER"
    
    # Check and create DeepSpeed configs if needed
    if [ ! -f "$PROPOSER_DEEPSPEED_CONFIG" ]; then
        create_deepspeed_config "$PROPOSER_DEEPSPEED_CONFIG" "proposer"
    fi
    
    if [ ! -f "$SOLVER_DEEPSPEED_CONFIG" ]; then
        create_deepspeed_config "$SOLVER_DEEPSPEED_CONFIG" "solver"
    fi
    
    # Check Python environment
    if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_DEFAULT_ENV:-}" ]]; then
        log_warn "No Python virtual environment detected."
        log_warn "Consider activating a virtual environment before running this script."
    fi
    
    # Print configuration
    echo ""
    echo "======================================"
    echo "PasoDoble Training Configuration"
    echo "======================================"
    echo "Proposer Model Name: $PROPOSER_MODEL_NAME"
    echo "Solver Model Name: $SOLVER_MODEL_NAME"
    echo "Proposer DeepSpeed Config: $PROPOSER_DEEPSPEED_CONFIG"
    echo "Solver DeepSpeed Config: $SOLVER_DEEPSPEED_CONFIG"
    echo "Wandb Run Name: $WANDB_RUN_NAME"
    echo "Output Directory Proposer: $OUTPUT_DIR_PROPOSER"
    echo "Output Directory Solver: $OUTPUT_DIR_SOLVER"
    echo "Max Steps: $MAX_STEPS"
    echo "Save Steps: $SAVE_STEPS"
    echo "Proposer Temperature: $PROPOSER_TEMPERATURE"
    echo "Solver Temperature: $SOLVER_TEMPERATURE"
    echo "Passing Rate Lower Threshold: $PASSING_RATE_LOWER_THRESHOLD"
    echo "Passing Rate Upper Threshold: $PASSING_RATE_UPPER_THRESHOLD"
    echo "Proposer VLLM Server Host: $PROPOSER_VLLM_SERVER_HOST"
    echo "Proposer VLLM Server Port: $PROPOSER_VLLM_SERVER_PORT"
    echo "Proposer VLLM Server Host: $PROPOSER_VLLM_CLIENT_PORT"
    echo "Solver VLLM Server Host: $SOLVER_VLLM_SERVER_HOST"
    echo "Solver VLLM Server Port: $SOLVER_VLLM_SERVER_PORT"
    echo "Solver VLLM Server Host: $SOLVER_VLLM_CLIENT_PORT"
    echo "GPU Memory Utilization: $VLLM_GPU_MEMORY_UTILIZATION"
    echo "Use Knowledge: $USE_KNOWLEDGE"
    echo "Use DeepSpeed: $USE_DS"
    echo "Use VLLM: $USE_VLLM"
    echo "Loss Type: $LOSS_TYPE"
    echo "Beta: $BETA"
    echo "======================================"
    echo ""
    
    # Start VLLM servers if needed
    if ! start_vllm_servers; then
        log_error "Failed to start VLLM servers"
        exit 1
    fi
    
    # Brief wait for servers to fully stabilize
    if [ "$USE_VLLM" -eq 1 ]; then
        log_info "Allowing servers to fully stabilize..."
        sleep 10
    fi
    
    # Run the training script
    log_info "Starting PasoDoble training..."
    
    # Determine accelerate config argument
    ACCELERATE_CONFIG_ARG=""
    if [ -f "configs/accelerate_config_6_cards.yaml" ]; then
        ACCELERATE_CONFIG_ARG="--config_file configs/accelerate_config_6_cards.yaml"
    fi
    
    export CUBLAS_WORKSPACE_CONFIG=:4096:8

    # Execute training command
    NCCL_DEBUG=INFO TORCH_NCCL_TRACE_BUFFER_SIZE="100" CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" accelerate launch $ACCELERATE_CONFIG_ARG \
        train_online.py \
        --proposer_model_name "$PROPOSER_MODEL_NAME" \
        --solver_model_name "$SOLVER_MODEL_NAME" \
        --proposer_deepspeed_path "$PROPOSER_DEEPSPEED_CONFIG" \
        --solver_deepspeed_path "$SOLVER_DEEPSPEED_CONFIG" \
        --wandb_run_name "$WANDB_RUN_NAME" \
        --output_dir_proposer "$OUTPUT_DIR_PROPOSER" \
        --output_dir_solver "$OUTPUT_DIR_SOLVER" \
        --max_steps "$MAX_STEPS" \
        --max_proposer_prompt_length "$MAX_PROPOSER_PROMPT_LENGTH" \
        --max_solver_prompt_length "$MAX_SOLVER_PROMPT_LENGTH" \
        --max_proposer_completion_length "$MAX_PROPOSER_COMPLETION_LENGTH" \
        --max_solver_completion_length "$MAX_SOLVER_COMPLETION_LENGTH" \
        --proposer_num_generations "$PROPOSER_NUM_GENERATIONS" \
        --solver_num_generations "$SOLVER_NUM_GENERATIONS" \
        --save_steps "$SAVE_STEPS" \
        --proposer_temperature "$PROPOSER_TEMPERATURE" \
        --solver_temperature "$SOLVER_TEMPERATURE" \
        --passing_rate_lower_threshold "$PASSING_RATE_LOWER_THRESHOLD" \
        --passing_rate_upper_threshold "$PASSING_RATE_UPPER_THRESHOLD" \
        --use_vllm "$USE_VLLM" \
        --vllm_mode "$VLLM_MODE" \
        --proposer_vllm_server_host "$PROPOSER_VLLM_SERVER_HOST" \
        --proposer_vllm_server_port "$PROPOSER_VLLM_SERVER_PORT" \
        --proposer_vllm_client_port "$PROPOSER_VLLM_CLIENT_PORT" \
        --solver_vllm_server_host "$SOLVER_VLLM_SERVER_HOST" \
        --solver_vllm_server_port "$SOLVER_VLLM_SERVER_PORT" \
        --solver_vllm_client_port "$SOLVER_VLLM_CLIENT_PORT" \
        --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
        --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        --use_knowledge "$USE_KNOWLEDGE" \
        --use_ds "$USE_DS" \
        --loss_type "$LOSS_TYPE" \
        --beta "$BETA"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "Training completed successfully!"
        log_success "Proposer results saved to: $OUTPUT_DIR_PROPOSER"
        log_success "Solver results saved to: $OUTPUT_DIR_SOLVER"
    else
        log_error "Training failed with exit code: $exit_code"
        exit $exit_code
    fi
}

# Run main function
main "$@"