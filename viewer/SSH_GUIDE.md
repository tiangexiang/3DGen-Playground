# SSH Tunnel Guide for Remote Visualization

This guide explains how to visualize 3D Gaussian Splatting data from a remote server on your local machine.

## Quick Start

### Step 1: Set up SSH tunnel (on your local machine)

```bash
ssh -L 8000:localhost:8000 user@remote-server-hostname
```

This command:
- Connects to your remote server via SSH
- Forwards port 8000 from the remote server to your local machine
- Keeps the connection alive

### Step 2: Start the server (on the remote server)

Once logged into the remote server via SSH:

```bash
cd /path/to/3DGen-Playground
python viewer/serve.py 8000
```

Or if you have Python 3 explicitly:

```bash
python3 viewer/serve.py 8000
```

### Step 3: Open in your local browser

On your local machine, open:
```
http://localhost:8000/viewer/index.html
```

The viewer will load from the remote server, and you can browse and visualize the 3DGS data that lives on the remote machine!

## Configuration

### Setting the data path

You have two options:

#### Option 1: Using .env file (recommended)

On the remote server, create or edit `.env` file in the project root:

```bash
# In 3DGen-Playground/.env
GS_PATH=/path/to/your/gaussian/splatting/data
```

Then start the server as normal.

#### Option 2: Manual configuration in UI

1. Start the server without setting `GS_PATH`
2. Open the viewer in your browser
3. Use the UI to configure the data path manually

### Using different ports

If port 8000 is already in use:

**Local machine:**
```bash
ssh -L 9000:localhost:9000 user@remote-server
```

**Remote server:**
```bash
python viewer/serve.py 9000
```

**Local browser:**
```
http://localhost:9000/viewer/index.html
```

## Advanced Usage

### Multiple SSH tunnels

You can forward multiple ports for different projects:

```bash
ssh -L 8000:localhost:8000 -L 8001:localhost:8001 user@remote-server
```

Then run servers on different ports:
```bash
# Terminal 1
cd project1 && python viewer/serve.py 8000

# Terminal 2  
cd project2 && python viewer/serve.py 8001
```

### Binding to all network interfaces

If you're on a trusted network and want to access the server from other machines without SSH:

```bash
python viewer/serve.py 8000 --host 0.0.0.0
```

Then access from any machine on the network:
```
http://server-ip-address:8000/viewer/index.html
```

⚠️ **Security Warning:** Only use `--host 0.0.0.0` on trusted networks, as it exposes the server to anyone on the network.

### Background server with tmux/screen

To keep the server running after disconnecting from SSH:

**Using tmux:**
```bash
tmux new -s viewer
python viewer/serve.py 8000
# Press Ctrl+B, then D to detach
# To reattach: tmux attach -t viewer
```

**Using screen:**
```bash
screen -S viewer
python viewer/serve.py 8000
# Press Ctrl+A, then D to detach
# To reattach: screen -r viewer
```

## Troubleshooting

### Port already in use

**Error:** `Port 8000 is already in use`

**Solution:** Use a different port:
```bash
python viewer/serve.py 8001
ssh -L 8001:localhost:8001 user@remote-server
```

### Connection refused

**Problem:** Browser shows "Connection refused"

**Checklist:**
1. ✓ Is the SSH tunnel active? (Check your SSH terminal)
2. ✓ Is the server running on the remote machine?
3. ✓ Are the ports matching on both sides?
4. ✓ Did you use `localhost` (not the remote hostname) in your browser?

### Data not loading

**Problem:** Viewer loads but 3DGS data doesn't appear

**Checklist:**
1. ✓ Is `GS_PATH` set correctly in `.env` or configured in UI?
2. ✓ Does the path exist on the remote server?
3. ✓ Do the data files have proper read permissions?
4. ✓ Check browser console (F12) for errors

### Firewall issues

If you can't connect even with SSH tunnel:

1. Check if the remote server firewall allows SSH: 
   ```bash
   sudo ufw status
   ```

2. Ensure SSH service is running:
   ```bash
   sudo systemctl status sshd
   ```

## Tips

- **Keep the SSH terminal open** - closing it will break the tunnel
- **Use absolute paths** for data outside the repo when setting `GS_PATH`
- **Check server logs** in the terminal for debugging
- **Use `screen` or `tmux`** for long-running visualization sessions
- **Forward multiple ports** if working with multiple datasets

## Example Workflow

Here's a complete example of a typical workflow:

```bash
# Local machine - establish tunnel
ssh -L 8000:localhost:8000 user@gpu-server.university.edu

# Remote server - navigate to project and set up data path
cd ~/projects/3DGen-Playground
echo 'GS_PATH=/data/gaussian-splatting-results' > .env

# Remote server - start the viewer server
python3 viewer/serve.py 8000

# Local browser - open the viewer
# Navigate to: http://localhost:8000/viewer/index.html
# Browse and visualize your 3DGS data interactively!
```

## Additional Resources

- SSH tunneling: `man ssh` (search for "LocalForward")
- Python SimpleHTTPServer documentation
- 3D Gaussian Splatting data format documentation

---

For more information, see the main [README.md](../README.md).

