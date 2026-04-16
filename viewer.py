import http.server
import socketserver
import webbrowser
import threading
from export_utils import export_obj_and_texture

def run_server():
    PORT = 8000
    try:
        Handler = http.server.SimpleHTTPRequestHandler
        # Use allow_reuse_address to avoid bind errors
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving 3D Viewer at http://localhost:{PORT}/viewer.html")
            httpd.serve_forever()
    except Exception as e:
        print(f"Server error or already running on port {PORT}: {e}")

def visualize_result(grid_x, grid_y, h_grid, color_grid):
    """
    Exports the generated textured saucer to an OBJ file and launches 
    an HTML5 Three.js 3D viewer in the browser with real-time reflections.
    """
    print("Exporting model to OBJ and texture to PNG...")
    export_obj_and_texture(grid_x, grid_y, h_grid, color_grid)
    
    print("Launching WebGL 3D Viewer...")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    webbrowser.open('http://localhost:8000/viewer.html')
    
    print("Press Enter to exit the viewer and terminate the script.")
    input()
