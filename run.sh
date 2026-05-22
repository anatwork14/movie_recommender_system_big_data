#!/bin/bash
# ==============================================================================
# Startup and Management Script — Hybrid Movie Recommender System
# ==============================================================================
# This script handles verification, building, data loading, and execution of the
# Recommender System, both in Docker (Containerized) and Local Dev modes.
# ==============================================================================

set -euo pipefail

# --- Color Definitions ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log_info() {
    echo -e "${BLUE}[INFO] $(date '+%H:%M:%S') - $1${NC}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $(date '+%H:%M:%S') - $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $(date '+%H:%M:%S') - $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $(date '+%H:%M:%S') - $1${NC}"
}

usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  up            Start the whole stack in Docker container mode (Recommended)"
    echo "  down          Stop and tear down the Docker container stack"
    echo "  local         Run the stack locally (starts Qdrant & Kafka in Docker, services locally)"
    echo "  index         Manually run the Qdrant indexer data pipeline"
    echo "  train-cf      Run PySpark ALS model training job"
    echo "  train-ncf     Run PyTorch Neural Collaborative Filtering training job"
    echo "  health        Check the status of running services"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo ""
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

COMMAND="$1"
shift

# --- Env setup ---
if [ ! -f .env ]; then
    log_info "Creating .env from template (.env.example)..."
    cp .env.example .env
fi

# Load environment variables (ignoring comments)
set -a
[ -f .env ] && . .env
set +a

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed."
        exit 1
    fi
}

check_node() {
    if ! command -v npm &> /dev/null; then
        log_error "Node/npm is not installed."
        exit 1
    fi
}

# --- Command Implementations ---

case "$COMMAND" in
    up)
        check_docker
        log_info "Starting Docker containers..."
        docker-compose up --build -d
        
        log_info "Waiting for Qdrant to be healthy..."
        for i in {1..30}; do
            if curl -s http://localhost:6333/healthz &> /dev/null; then
                log_success "Qdrant is ready!"
                break
            fi
            sleep 2
        done

        # Check if index has been populated in Qdrant
        log_info "Checking if Qdrant collection 'movie_content' exists..."
        INDEX_EXISTS=$(curl -s http://localhost:6333/collections | grep -q "movie_content" && echo "yes" || echo "no")
        if [ "$INDEX_EXISTS" = "no" ]; then
            log_warning "Qdrant index is empty. Running Qdrant Indexer Pipeline..."
            if command -v python3 &> /dev/null; then
                # Check for dependencies
                python3 -c "import qdrant_client, sentence_transformers, pandas" &> /dev/null || {
                    log_warning "Local python missing indexing requirements. Triggering inside docker..."
                    docker-compose exec -T recsys-backend python data_processing_pipeline/indexer.py
                    exit 0
                }
                python3 data_processing_pipeline/indexer.py
            else
                log_warning "Python3 not found locally. Triggering indexing inside recsys-backend container..."
                docker-compose exec -T recsys-backend python data_processing_pipeline/indexer.py
            fi
        else
            log_success "Qdrant index is already populated."
        fi

        log_success "Recommender System is fully running!"
        log_info "Frontend: http://localhost:3005"
        log_info "FastAPI Backend Docs: http://localhost:5001/docs"
        ;;

    down)
        check_docker
        log_info "Tearing down Docker containers..."
        docker-compose down
        log_success "All services stopped."
        ;;

    local)
        check_docker
        check_python
        check_node

        log_info "Starting external resources (Qdrant & Kafka) in background via Docker..."
        docker-compose up recsys-kafka recsys-qdrant -d

        log_info "Waiting for background services to reach healthy state..."
        for i in {1..30}; do
            if curl -s http://localhost:6333/healthz &> /dev/null; then
                break
            fi
            sleep 2
        done

        # Qdrant Index Check
        INDEX_EXISTS=$(curl -s http://localhost:6333/collections | grep -q "movie_content" && echo "yes" || echo "no")
        if [ "$INDEX_EXISTS" = "no" ]; then
            log_info "Running Qdrant data indexer..."
            python3 data_processing_pipeline/indexer.py
        fi

        # Install dependencies if node_modules doesn't exist
        if [ ! -d frontend/node_modules ]; then
            log_info "Installing frontend node dependencies..."
            cd frontend && npm install && cd ..
        fi

        log_info "Starting local FastAPI Backend (Port 5001)..."
        # Run backend in background
        python3 -m uvicorn backend.main:app --reload --port 5001 &
        BACKEND_PID=$!

        log_info "Starting local Vite Frontend (Port 5173)..."
        # Run frontend
        cd frontend
        npm run dev &
        FRONTEND_PID=$!
        cd ..

        # Handle process cleanup on exit
        trap 'kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true; exit' INT TERM EXIT
        
        log_success "All local services started successfully."
        log_info "Press Ctrl+C to terminate both servers."
        
        # Keep shell alive
        wait
        ;;

    index)
        check_python
        log_info "Executing data indexing pipeline into Qdrant..."
        python3 data_processing_pipeline/indexer.py
        log_success "Qdrant indexing complete!"
        ;;

    train-cf)
        check_python
        log_info "Running Spark ALS Collaborative Filtering model training job..."
        python3 spark_jobs/collaborative_filtering.py
        log_success "CF factor matrices saved to models/als/!"
        ;;

    train-ncf)
        check_python
        log_info "Running PyTorch Neural Collaborative Filtering training job..."
        python3 spark_jobs/ncf_training.py
        log_success "NCF weights and metadata saved to models/ncf/!"
        ;;

    health)
        log_info "--- System Health Check ---"
        if curl -s http://localhost:6333/healthz &> /dev/null; then
            echo -e "Qdrant: ${GREEN}Online (Healthy)${NC}"
        else
            echo -e "Qdrant: ${RED}Offline${NC}"
        fi

        if curl -fs http://localhost:5001/api/health &> /dev/null; then
            echo -e "FastAPI Backend: ${GREEN}Online${NC}"
            curl -s http://localhost:5001/api/health
        else
            echo -e "FastAPI Backend: ${RED}Offline${NC}"
        fi
        ;;

    *)
        usage
        ;;
esac
