#!/usr/bin/env python3
"""
Simple HTTP server for the 3DGen Playground Interactive Viewer.

This server handles CORS and proper MIME types for serving 3DGS data and the viewer.
Supports both Python 3.x versions and SSH tunneling for remote access.

Usage:
    python serve.py [port] [--host HOST]
    
Default port: 8000
Default host: localhost (127.0.0.1)

Examples:
    python serve.py 8080
    python serve.py 8000 --host 0.0.0.0  # Allow remote access (e.g., via SSH tunnel)
    
SSH Tunnel Usage (for remote servers):
    # On your local machine:
    ssh -L 8000:localhost:8000 user@remote-server
    
    # On the remote server:
    python serve.py 8000
    
    # Then open http://localhost:8000/viewer/index.html in your local browser
"""

import http.server
import socketserver
import sys
import os
import json
import mimetypes
import argparse
import socket
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote


def load_env_file():
    """Load environment variables from .env file."""
    env_vars = {}
    env_file = Path(__file__).parent.parent / '.env'
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    
    return env_vars


def normalize_web_path(path, project_root):
    """
    Normalize a path for web serving.
    
    This function handles:
    - Tilde paths (~/) -> expand to absolute path
    - Absolute filesystem paths inside the repo -> convert to relative web path
    - Absolute filesystem paths outside the repo -> mark with special prefix
    - Relative paths -> ensure they're relative to repo root
    
    Examples:
        'sample_data' -> 'sample_data'
        '../sample_data' -> 'sample_data'
        '~/test/data' -> '/absolute/home/user/test/data' (expanded and outside repo)
        '/Users/user/repo/sample_data' -> 'sample_data' (inside repo)
        '/other/path/data' -> '/absolute/other/path/data' (outside repo, special prefix)
    """
    path = path.rstrip('/')  # Remove trailing slashes
    
    # Expand tilde (~) to home directory first, before any other processing
    path = os.path.expanduser(path)
    
    # If it's an absolute filesystem path
    if path.startswith('/'):
        # Convert to Path for proper path operations
        abs_path = Path(path).resolve()
        repo_root = Path(project_root).resolve()
        
        # Check if the absolute path is inside the repository
        try:
            # Get relative path from repo root
            rel_path = abs_path.relative_to(repo_root)
            return str(rel_path)
        except ValueError:
            # Path is outside the repository - use special /absolute/ prefix
            # This will be handled by custom endpoint
            return f'/absolute{abs_path}'
    
    # If path starts with ../, remove one level since viewer is in viewer/
    # and we want paths relative to repo root
    if path.startswith('../'):
        path = path[3:]  # Remove '../'
    
    return path


# Load environment variables
ENV_VARS = load_env_file()


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support and proper MIME types."""
    
    def __init__(self, *args, **kwargs):
        # Set the directory to serve from (project root)
        project_root = Path(__file__).parent.parent
        super().__init__(*args, directory=str(project_root), **kwargs)
    
    def end_headers(self):
        """Add CORS headers to allow cross-origin requests."""
        # Enable CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        
        # Cache control for better performance
        if self.path.endswith(('.ply', '.spz', '.splat')):
            self.send_header('Cache-Control', 'public, max-age=3600')
        
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight."""
        self.send_response(200)
        self.end_headers()
    
    def do_HEAD(self):
        """Handle HEAD requests - delegate to GET handler."""
        self.do_GET()
    
    def do_GET(self):
        """Handle GET requests with custom API endpoints."""
        parsed_path = urlparse(self.path)
        
        # API endpoint to get configuration
        if parsed_path.path == '/api/config':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get GS_PATH from environment
            gs_path = ENV_VARS.get('GS_PATH')
            
            # Normalize path if present
            if gs_path:
                project_root = Path(__file__).parent.parent
                gs_path = normalize_web_path(gs_path, project_root)
            
            config = {
                'dataRootPath': gs_path,
                'source': 'env' if gs_path else 'none'
            }
            
            self.wfile.write(json.dumps(config).encode())
            return
        
        # Handle absolute filesystem paths (outside repo)
        # These start with /absolute/ prefix
        if parsed_path.path.startswith('/absolute/'):
            # Extract the actual filesystem path
            fs_path = parsed_path.path[9:]  # Remove '/absolute' prefix
            fs_path = unquote(fs_path)  # Decode URL encoding
            
            try:
                file_path = Path(fs_path)
                
                # Security check - ensure file exists and is readable
                if not file_path.exists():
                    self.send_error(404, f"File not found: {fs_path}")
                    return
                
                if not file_path.is_file():
                    self.send_error(403, "Path is not a file")
                    return
                
                # Read and serve the file
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                # Determine content type
                content_type = self.guess_type(str(file_path))
                
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.send_header('Content-Length', len(content))
                self.end_headers()
                self.wfile.write(content)
                return
                
            except Exception as e:
                self.send_error(500, f"Error reading file: {str(e)}")
                return
        
        # Default file serving for paths within repo
        super().do_GET()
    
    def guess_type(self, path):
        """Override to add custom MIME types for 3DGS files."""
        mimetype = super().guess_type(path)
        
        # Add custom MIME types for 3D formats
        if path.endswith('.ply'):
            return 'application/octet-stream'
        elif path.endswith('.spz'):
            return 'application/octet-stream'
        elif path.endswith('.splat'):
            return 'application/octet-stream'
        elif path.endswith('.ksplat'):
            return 'application/octet-stream'
        
        return mimetype
    
    def log_message(self, format, *args):
        """Custom log format with colors."""
        # Color codes
        GREEN = '\033[92m'
        BLUE = '\033[94m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        RESET = '\033[0m'
        
        # Get status code
        status_code = args[1] if len(args) > 1 else '000'
        
        # Choose color based on status code
        if status_code.startswith('2'):
            color = GREEN
        elif status_code.startswith('3'):
            color = BLUE
        elif status_code.startswith('4'):
            color = YELLOW
        else:
            color = RED
        
        # Log with color
        sys.stderr.write(f"{color}[{self.log_date_time_string()}] {format % args}{RESET}\n")


def get_hostname():
    """Get the hostname of the current machine."""
    try:
        return socket.gethostname()
    except:
        return "unknown"


def main():
    """Start the development server."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='3DGen Playground Interactive Viewer Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python serve.py                    # Start on localhost:8000
  python serve.py 8080               # Start on localhost:8080
  python serve.py 8000 --host 0.0.0.0  # Allow remote access
  
SSH Tunnel (for remote servers):
  # On local machine:
  ssh -L 8000:localhost:8000 user@remote-server
  
  # On remote server:
  python serve.py 8000
  
  # Then visit http://localhost:8000/viewer/index.html locally
        """
    )
    parser.add_argument('port', type=int, nargs='?', default=8000,
                        help='Port to run the server on (default: 8000)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to bind to (default: 127.0.0.1). Use 0.0.0.0 for all interfaces')
    
    args = parser.parse_args()
    port = args.port
    host = args.host
    
    # Ensure we're in the correct directory
    viewer_dir = Path(__file__).parent
    project_root = viewer_dir.parent
    
    # Display configuration
    gs_path = ENV_VARS.get('GS_PATH')
    normalized_path = None
    
    if gs_path:
        normalized_path = normalize_web_path(gs_path, project_root)
        env_status = '✓ From .env'
    else:
        env_status = '⚠️  Not configured (set GS_PATH in .env or configure in UI)'
    
    # Get hostname for display
    hostname = get_hostname()
    
    print("\n" + "="*70)
    print("🎨 3DGen Playground - Interactive 3DGS Viewer")
    print("="*70)
    print(f"\n📂 Serving from: {project_root}")
    print(f"🌐 Server binding: {host}:{port}")
    
    # Show appropriate URLs based on host binding
    if host == '0.0.0.0':
        print(f"\n🔗 Access URLs:")
        print(f"   Local:        http://localhost:{port}/viewer/index.html")
        print(f"   Network:      http://{hostname}:{port}/viewer/index.html")
        print(f"   (or use your machine's IP address)")
    elif host == '127.0.0.1' or host == 'localhost':
        print(f"🔗 Viewer URL:   http://localhost:{port}/viewer/index.html")
    else:
        print(f"🔗 Viewer URL:   http://{host}:{port}/viewer/index.html")
    
    print(f"\n⚙️  Configuration:")
    if gs_path:
        print(f"   GS_PATH (.env): {gs_path}")
        if normalized_path != gs_path:
            if normalized_path.startswith('/absolute/'):
                print(f"   Type: Absolute path (outside repo)")
                print(f"   Web path: {normalized_path}")
            else:
                print(f"   Type: Relative path (inside repo)")
                print(f"   Web path: {normalized_path}")
    else:
        print(f"   GS_PATH: (not set)")
    print(f"   Status: {env_status}")
    
    # Show SSH tunnel instructions if on localhost
    if host in ['127.0.0.1', 'localhost']:
        print("\n🔐 SSH Tunnel (for remote servers):")
        print(f"   On local:  ssh -L {port}:localhost:{port} user@{hostname}")
        print(f"   On remote: python serve.py {port}")
        print(f"   Then open: http://localhost:{port}/viewer/index.html")
    
    print("\n💡 Tips:")
    print("   - GS_PATH can be relative ('sample_data') or absolute ('/path/to/data')")
    print("   - Configure data path in .env or manually in the viewer UI")
    if host in ['127.0.0.1', 'localhost']:
        print("   - Use --host 0.0.0.0 to allow remote network access")
    print("   - Press Ctrl+C to stop the server")
    print("\n" + "="*70 + "\n")
    
    # Create server
    try:
        with socketserver.TCPServer((host, port), CORSRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!\n")
        sys.exit(0)
    except OSError as e:
        if e.errno == 48 or e.errno == 98:  # Address already in use
            print(f"\n❌ Error: Port {port} is already in use.")
            print(f"   Try a different port: python serve.py {port + 1}\n")
            sys.exit(1)
        else:
            raise


if __name__ == "__main__":
    main()

