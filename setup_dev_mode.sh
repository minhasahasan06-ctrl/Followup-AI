#!/bin/bash

echo "🔧 Setting up Development Mode for Python Backend"
echo ""

# Generate secure dev mode secret
DEV_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

echo "✅ Generated secure DEV_MODE_SECRET"
echo ""
echo "Add this to your Replit Secrets:"
echo "  Key: DEV_MODE_SECRET"
echo "  Value: $DEV_SECRET"
echo ""
echo "Or set it as environment variable:"
echo "  export DEV_MODE_SECRET=\"$DEV_SECRET\""
echo ""
echo "Then restart the Python backend with:"
echo "  python3 start_python_server.py"
