# TradingAgents

Multi-Agents LLM Financial Trading Framework

## Description

TradingAgents is a sophisticated multi-agent LLM framework designed for financial trading and portfolio management. It leverages AI-powered agents to analyze markets, manage portfolios, and execute trading strategies.

## Features

- Multi-agent architecture for trading analysis
- Portfolio management capabilities
- Integration with Alpaca trading API
- Technical and fundamental analysis tools
- CLI interface for easy interaction
- Automated trading strategies

## Installation

```bash
poetry install
```

## Configuration

### Required Environment Variables

The following environment variables are **REQUIRED** for both local and production deployments:

#### AWS S3 Configuration (Required)

```bash
export S3_BUCKET_NAME=mintrader-reports  # Your S3 bucket name
export S3_REGION=us-east-1               # AWS region
export AWS_ACCESS_KEY_ID=your_key        # AWS access key
export AWS_SECRET_ACCESS_KEY=your_secret # AWS secret key
```

**Note:** S3 is required for state management and continuity. The system always fetches from and uploads to S3, even when running locally.

#### Alpaca Trading API (Required)

```bash
export ALPACA_API_KEY=your_alpaca_key
export ALPACA_SECRET_KEY=your_alpaca_secret
```

#### Optional: LLM Configuration

```bash
# For production (OpenAI)
export OPENAI_API_KEY=your_openai_key
export LLM_MODEL=gpt-4o-mini

# For local development (Ollama - auto-detected)
# Just run: ollama serve
```

## Usage

### Trading Agents CLI

```bash
tradingagents
```

### Portfolio Manager

```bash
portfoliomanager
```

## Requirements

- Python ^3.10
- AWS S3 bucket with proper credentials (REQUIRED)
- Alpaca trading account (paper or live)
- See `pyproject.toml` for full dependency list

## License

Apache-2.0

## Repository

https://github.com/TauricResearch/TradingAgents
