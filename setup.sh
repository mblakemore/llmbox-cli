#!/usr/bin/env bash
set -e

echo "=== llmbox setup ==="
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "  Created .venv/"
else
    echo "Virtual environment already exists."
fi

# Install Python dependencies
echo "Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet requests markdownify PyMuPDF prompt_toolkit
echo "  Done."

# Install gateway dependencies (optional, for cc_gateway.py)
echo "Installing gateway dependencies..."
"$VENV_DIR/bin/pip" install --quiet fastapi uvicorn
echo "  Done."

# Make llmbox.sh executable
chmod +x "$SCRIPT_DIR/llmbox.sh"

# Symlink to ~/.local/bin
mkdir -p ~/.local/bin
ln -sf "$SCRIPT_DIR/llmbox.sh" ~/.local/bin/llmbox
echo "  Installed 'llmbox' command to ~/.local/bin/llmbox"

# Check PATH
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    SHELL_RC=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
    fi
    if [ -n "$SHELL_RC" ]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        echo "  Added ~/.local/bin to PATH in $SHELL_RC"
        echo "  Run: source $SHELL_RC"
    else
        echo "  NOTE: Add ~/.local/bin to your PATH:"
        echo '    export PATH="$HOME/.local/bin:$PATH"'
    fi
fi

# Prompt for API config if not set
echo
if [ -z "$BEDROCK_API_URL" ] || [ -z "$BEDROCK_API_KEY" ]; then
    echo "Configure API access (or press Enter to skip and set later):"
    read -rp "  BEDROCK_API_URL: " API_URL
    read -rp "  BEDROCK_API_KEY: " API_KEY

    if [ -n "$API_URL" ] && [ -n "$API_KEY" ]; then
        SHELL_RC=""
        if [ -f "$HOME/.zshrc" ]; then
            SHELL_RC="$HOME/.zshrc"
        elif [ -f "$HOME/.bashrc" ]; then
            SHELL_RC="$HOME/.bashrc"
        fi
        if [ -n "$SHELL_RC" ]; then
            echo "" >> "$SHELL_RC"
            echo "# llmbox" >> "$SHELL_RC"
            echo "export BEDROCK_API_URL=\"$API_URL\"" >> "$SHELL_RC"
            echo "export BEDROCK_API_KEY=\"$API_KEY\"" >> "$SHELL_RC"
            echo "  Saved to $SHELL_RC"
            echo "  Run: source $SHELL_RC"
        else
            echo "  Add these to your shell profile:"
            echo "    export BEDROCK_API_URL=\"$API_URL\""
            echo "    export BEDROCK_API_KEY=\"$API_KEY\""
        fi
    else
        echo "  Skipped. Set BEDROCK_API_URL and BEDROCK_API_KEY before running."
    fi
else
    echo "API config already set."
fi

echo
echo "=== Setup complete ==="
echo "Run 'llmbox' from any directory to start."
