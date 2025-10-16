# Interactive 3DGS Viewer

A production-ready, developer-friendly web viewer for exploring 3D Gaussian Splatting (3DGS) data from the 3DGen Playground dataset.

![3DGS Viewer](https://img.shields.io/badge/Viewer-Interactive-ff6c6c?style=flat-square) ![Powered by Spark](https://img.shields.io/badge/Powered%20by-Spark-blue?style=flat-square)

## Prerequisites

The viewer requires [Spark](https://github.com/sparkjsdev/spark), which is included as a git submodule. If you cloned the repository without submodules, initialize it:

```bash
# From repository root
git submodule update --init --recursive
```

## Features

- 🎯 **Interactive Input**: Load any 3DGS object by entering its ID or path
- ⚡ **Real-time Rendering**: Powered by [Spark](https://sparkjs.dev) for high-quality 3DGS visualization
- 🎨 **Modern UI**: Clean, intuitive interface with responsive design
- 🔧 **Configurable**: Easy configuration via JSON for different data locations
- 📱 **Responsive**: Works on desktop and mobile devices
- 🚀 **Fast Loading**: Optimized loading with visual feedback

## Quick Start

### 1. Start the Development Server

From the repository root, run:

```bash
cd viewer
python serve.py
```

Or specify a custom port:

```bash
python serve.py 8080
```

### 2. Open the Viewer

Navigate to: **http://localhost:8000/viewer/index.html**

### 3. Configure Data Root Path (Optional)

The viewer automatically reads `GS_PATH` from your `.env` file! If you've already set up `.env` for downloading data, you're all set.

**To override the .env setting:**
- In the viewer interface, modify the **Data Root** field
- Click **Save** to persist your custom path
- The override is saved in browser localStorage

### 4. Load 3DGS Data

You can load objects in three ways:

#### Option A: Use the Input Box
Type an object ID (e.g., `9350303`) or path (e.g., `1876/9374307`) and click "Load"

#### Option B: Click Quick Examples
Click any of the pre-configured example chips below the input box

#### Option C: Use URL Parameters
Open the viewer with a specific object:
```
http://localhost:8000/viewer/index.html?object=9350303
```

## Configuration

### Data Root Path Priority

The viewer uses this priority system for finding your 3DGS data:

1. **Browser Override** (localStorage) - if you've manually saved a path in the UI
2. **GS_PATH from .env** - automatically read from your project's `.env` file
3. **Default Fallback** - `../sample_data` if nothing else is configured

### Setting Up .env (Recommended)

If you haven't already, configure your `.env` file at the repository root:

```bash
# In the repository root
cp .env.example .env

# Edit .env and set GS_PATH
# Example:
GS_PATH=/path/to/your/gaussianverse/aesthetic
```

The viewer will automatically use this path! This is consistent with how the data download scripts work.

### Manual Override

If you need to temporarily use a different path:

1. Open the viewer
2. Change the **Data Root** field
3. Click **Save**
4. The override persists in your browser

To revert to `.env` settings, clear browser localStorage or delete the override.

## Data Structure

The viewer expects data organized as follows:

```
data_root/
├── 9350303/
│   ├── point_cloud.ply    # Required: 3DGS data
│   ├── cameras.json       # Optional: camera metadata
│   └── exposure.json      # Optional: exposure settings
├── 9351397/
│   └── point_cloud.ply
└── 1876/
    └── 9374307/
        └── point_cloud.ply
```

The viewer loads `{dataRootPath}/{objectId}/point_cloud.ply`

## Usage Examples

### Example 1: View Sample Data

Using the included sample data:

```bash
# Start server
python serve.py

# Open browser to http://localhost:8000/viewer/index.html
# Data Root: ../sample_data (default)
# Object Path: 9350303
# Click "Load"
```

### Example 2: View Downloaded GaussianVerse Data

1. Download GaussianVerse data (see [data/README.md](../data/README.md))
2. Your `.env` file should have:
   ```bash
   GS_PATH=/path/to/gaussianverse/aesthetic
   ```
3. Start the viewer - it automatically uses the path from `.env`
4. Enter object paths like: `1876/9374307`, `2000/9800123`, etc.
5. Click **Load**

### Example 3: Direct URL Access

Share specific objects with URL parameters:

```
http://localhost:8000/viewer/index.html?object=9585727
```

## Controls

- **Left Mouse**: Rotate camera
- **Right Mouse / Two Fingers**: Pan camera
- **Scroll / Pinch**: Zoom in/out

## Troubleshooting

### Issue: "File not found" error

**Solutions:**
- Verify the object ID exists in your data directory
- Check that `GS_PATH` in `.env` points to the correct directory
- Ensure `point_cloud.ply` exists in the object folder
- Check file permissions (files should be readable)
- Look at the server console to see what path is being used

### Issue: Viewer not loading

**Solutions:**
- Make sure you're using the development server (`python serve.py`)
- Don't open `index.html` directly in the browser (CORS issues)
- Check browser console for errors (F12)
- Verify Spark library is present in `../spark/dist/`

### Issue: Port already in use

**Solution:**
```bash
python serve.py 8080  # Try a different port
```

### Issue: Slow loading

**Possible causes:**
- Large PLY files (some objects can be 100MB+)
- Network issues (if loading from URL)
- Browser performance (try Chrome/Edge for best results)

## Technical Details

### Architecture

```
viewer/
├── index.html       # Main viewer interface
├── config.json      # Configuration file
├── serve.py         # Development server
└── README.md        # This file
```

### Dependencies

- **Spark**: 3DGS rendering engine (included in `../spark/`)
- **THREE.js**: 3D graphics library (loaded via CDN)
- **Python 3.x**: For development server

### Browser Compatibility

- ✅ Chrome/Edge (Recommended)
- ✅ Firefox
- ✅ Safari (macOS/iOS)
- ⚠️ Requires WebGL 2.0 support

## Development

### Customizing the Viewer

The viewer is built with vanilla JavaScript and can be easily customized:

- **Styles**: Edit the `<style>` section in `index.html`
- **Behavior**: Modify the `<script type="module">` section
- **Configuration**: Add new options to `config.json`

### Adding New Features

Example: Add a screenshot button

```javascript
// In index.html, add button to HTML
<button id="screenshotBtn">📸 Screenshot</button>

// Add event handler in JavaScript
document.getElementById('screenshotBtn').addEventListener('click', () => {
  renderer.domElement.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'screenshot.png';
    a.click();
  });
});
```

## Integration with Training Pipelines

The viewer can be integrated with your training/evaluation pipelines:

```python
# In your training script
def visualize_generated_3dgs(output_path, object_id):
    """Open the viewer for a generated 3DGS object."""
    import webbrowser
    viewer_url = f"http://localhost:8000/viewer/index.html?object={object_id}"
    webbrowser.open(viewer_url)

# After generating a 3DGS
save_point_cloud(model.generate(), f"outputs/{epoch}/point_cloud.ply")
visualize_generated_3dgs("outputs", f"{epoch}")
```

## FAQ

**Q: Can I use this in production?**  
A: The viewer is production-ready, but use a proper web server (nginx, Apache) instead of `serve.py` for production deployments.

**Q: Does it work offline?**  
A: Yes, but you need to download THREE.js locally. Update the import map in `index.html` to point to local files.

**Q: Can I embed this in my application?**  
A: Yes! The viewer can be embedded in an iframe or integrated into larger web applications.

**Q: How do I handle authentication?**  
A: Modify `serve.py` to add authentication middleware or use a reverse proxy with auth.

## Contributing

Contributions are welcome! Please:

1. Test your changes with multiple objects
2. Ensure cross-browser compatibility
3. Update this README if adding new features
4. Follow the existing code style

## License

This viewer is part of the 3DGen Playground project. See the main repository LICENSE file.

## Acknowledgments

- Powered by [Spark](https://sparkjs.dev) from [World Labs](https://www.worldlabs.ai/)
- Uses [GaussianVerse](https://gaussianverse.stanford.edu) data format
- Built with [THREE.js](https://threejs.org)

## Support

For issues or questions:
- Open an issue on the main repository
- Check the [Spark documentation](https://sparkjs.dev)
- Review the [3DGen Playground README](../README.md)

---

**Happy Viewing! 🎨✨**

