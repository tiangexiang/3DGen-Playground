#!/usr/bin/env python3
"""
Simple HTTP server for the 3DGen Playground Interactive Viewer.

This server handles CORS and proper MIME types for serving 3DGS data and the viewer.
Supports both Python 3.x versions.

Usage:
    python serve.py [port]
    
Default port: 8000

Example:
    python serve.py 8080
"""

import http.server
import socketserver
import sys
import os
import json
import mimetypes
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
    - Absolute filesystem paths inside the repo -> convert to relative web path
    - Absolute filesystem paths outside the repo -> mark with special prefix
    - Relative paths -> ensure they're relative to repo root
    
    Examples:
        'sample_data' -> 'sample_data'
        '../sample_data' -> 'sample_data'
        '/Users/user/repo/sample_data' -> 'sample_data' (inside repo)
        '/other/path/data' -> '/absolute/other/path/data' (outside repo, special prefix)
    """
    path = path.rstrip('/')  # Remove trailing slashes
    
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


def main():
    """Start the development server."""
    # Get port from command line or use default
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
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
    
    print("\n" + "="*60)
    print("🎨 3DGen Playground - Interactive 3DGS Viewer")
    print("="*60)
    print(f"\n📂 Serving from: {project_root}")
    print(f"🌐 Server running at: http://localhost:{port}")
    print(f"🔗 Viewer URL: http://localhost:{port}/viewer/index.html")
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
    print("\n💡 Tips:")
    print("   - GS_PATH can be relative ('sample_data') or absolute ('/path/to/data')")
    print("   - Or configure manually in the viewer UI")
    print("   - Press Ctrl+C to stop the server")
    print("\n" + "="*60 + "\n")
    
    # Create server
    try:
        with socketserver.TCPServer(("", port), CORSRequestHandler) as httpd:
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

