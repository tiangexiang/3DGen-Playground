import os
import json
import argparse
import logging
import time
import random
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

download_url_prefix = 'https://downloads.cs.stanford.edu/vision/gaussianverse/'

def download_url(url, save_dir, skip_existing=True, max_retries=5, base_wait=1.0, worker_idx=0, stagger_delay=10):
    """
    Download a file from URL to save_dir with progress display and retry logic.
    
    Args:
        url: The URL to download from
        save_dir: Directory to save the file
        skip_existing: If True, skip download if file already exists
        max_retries: Maximum number of retry attempts for 503 errors
        base_wait: Base wait time in seconds (will use exponential backoff)
        worker_idx: Worker index (0-based) for staggered start
        stagger_delay: Delay in seconds per worker index (worker waits stagger_delay * worker_idx)
        
    Returns:
        dict: Status information about the download
    """
    try:
        # Staggered start: each worker waits based on its index
        initial_wait = stagger_delay * worker_idx
        if initial_wait > 0:
            logger.info(f"Worker {worker_idx}: waiting {initial_wait}s before starting...")
            time.sleep(initial_wait)
        
        # Add a small random delay to avoid overwhelming the server
        time.sleep(random.uniform(1.0, 3.0))
        
        # Parse URL to get filename
        filename = url.split('/')[-1]
        
        # Create target directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Full path to the file
        target_file = os.path.join(save_dir, filename)
        
        # Check if file already exists
        if skip_existing and os.path.exists(target_file):
            file_size = os.path.getsize(target_file)
            if file_size > 0:  # Only skip if file is not empty
                logger.info(f"Worker {worker_idx}: ⊘ Skipping {filename} (already exists, {file_size:,} bytes)")
                return {
                    'status': 'skipped',
                    'url': url,
                    'filename': filename,
                    'reason': 'already_exists'
                }
        
        # Retry loop for handling 503 and other transient errors
        last_error = None
        for attempt in range(max_retries):
            if attempt > 0:
                # Exponential backoff with jitter for retries
                wait_time = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 2)
                logger.warning(f"Worker {worker_idx}: ⟳ Retry {attempt + 1}/{max_retries} for {filename} after {wait_time:.1f}s wait...")
                time.sleep(wait_time)
            
            logger.info(f"Worker {worker_idx}: ↓ Downloading {filename}...")
            
            # Download using wget with minimal output for parallel downloads
            command = [
                'wget',
                '--tries=1',  # We handle retries in the loop
                '--timeout=120',  # 2 minute timeout for large files
                '--continue',  # Continue partial downloads
                '--no-verbose',  # Suppress verbose output for cleaner parallel downloads
                '--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                '--header=Accept: */*',
                '--header=Connection: keep-alive',
                '-O', target_file,
                url
            ]
            
            # Run wget and capture output
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                if os.path.exists(target_file):
                    file_size = os.path.getsize(target_file)
                    logger.info(f"Worker {worker_idx}: ✓ Successfully downloaded {filename} ({file_size:,} bytes)")
                    return {
                        'status': 'success',
                        'url': url,
                        'filename': filename,
                        'size': file_size
                    }
            
            # Get error message from wget output
            error_message = result.stderr.strip() if result.stderr else ""
            last_error = error_message
            
            # Check if it's a 503 error
            if '503' in error_message or 'Service Temporarily Unavailable' in error_message:
                if attempt < max_retries - 1:
                    logger.warning(f"Worker {worker_idx}: ✗ Got 503 Service Unavailable for {filename}")
                    continue
                else:
                    logger.error(f"Worker {worker_idx}: ✗ Failed to download {filename} after {max_retries} attempts: 503 Service Unavailable")
            else:
                if attempt < max_retries - 1:
                    logger.warning(f"Worker {worker_idx}: ✗ Download failed for {filename}")
                    continue
                else:
                    logger.error(f"Worker {worker_idx}: ✗ Failed to download {filename} after {max_retries} attempts")
        
        # All retries exhausted
        return {
            'status': 'failed',
            'url': url,
            'filename': filename,
            'error': last_error or 'Max retries exceeded'
        }
            
    except Exception as e:
        logger.error(f"✗ Error downloading {url}: {str(e)}")
        return {
            'status': 'error',
            'url': url,
            'error': str(e)
        }


def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download GaussianVerse 3D Gaussian Splatting files.')
    parser.add_argument('--save_dir', type=str, required=True, 
                        help='Directory to save downloaded files')
    parser.add_argument('--download_meta', type=str2bool, default=True,
                        help='Download meta files (mean, std, sphere2plane, lists) (default: True)')
    parser.add_argument('--download_aesthetic', type=str2bool, default=True,
                        help='Download aesthetic chunk files (default: True)')
    parser.add_argument('--download_non_aesthetic', type=str2bool, default=True,
                        help='Download non-aesthetic chunk files (default: True)')
    parser.add_argument('--workers', type=int, default=None,
                        help=f'Number of parallel download workers (default: auto - uses all {os.cpu_count()} CPU cores, capped at number of files)')
    parser.add_argument('--stagger-delay', type=int, default=2,
                        help='Delay in seconds between worker starts (worker N waits N * stagger_delay, default: 2)')
    parser.add_argument('--max-retries', type=int, default=5,
                        help='Maximum retry attempts per file (default: 5)')
    parser.add_argument('--base-wait', type=float, default=3.0,
                        help='Base wait time in seconds for exponential backoff (default: 3.0)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip files that already exist (default: True)')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                        help='Re-download all files even if they exist')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose debug logging')

    args = parser.parse_args()
    
    # Check that at least one download type is enabled
    if not (args.download_meta or args.download_aesthetic or args.download_non_aesthetic):
        logger.error("Error: At least one download type must be enabled.")
        logger.error("All download flags are set to False. Please set at least one to True:")
        logger.error("  --download_meta True")
        logger.error("  --download_aesthetic True")
        logger.error("  --download_non_aesthetic True")
        exit(1)
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Build file list based on flags
    files_to_download = []
    
    if args.download_meta:
        meta_files = [
            'gaussianverse_mean.pt',
            'gaussianverse_std.pt',
            'sphere2plane.npy',
            'aesthetic_list.json',
            'non_aesthetic_list.json',
            'all_obj_list.json'
        ]
        files_to_download.extend(meta_files)
        logger.info(f"Added {len(meta_files)} meta files to download queue")
    
    if args.download_aesthetic:
        # GaussianVerse_aesthetic_chunk_0.zip through GaussianVerse_aesthetic_chunk_8.zip
        aesthetic_files = [f'GaussianVerse_aesthetic_chunk_{i}.zip' for i in range(9)]
        files_to_download.extend(aesthetic_files)
        logger.info(f"Added {len(aesthetic_files)} aesthetic chunk files to download queue")
    
    if args.download_non_aesthetic:
        # GaussianVerse_chunk_0.zip through GaussianVerse_chunk_17.zip
        non_aesthetic_files = [f'GaussianVerse_chunk_{i}.zip' for i in range(18)]
        files_to_download.extend(non_aesthetic_files)
        logger.info(f"Added {len(non_aesthetic_files)} non-aesthetic chunk files to download queue")
    
    total_urls = len(files_to_download)
    
    # Determine number of workers: use all CPU cores, capped at number of files
    if args.workers is None:
        cpu_count = os.cpu_count() or 4  # Fallback to 4 if cpu_count returns None
        num_workers = min(cpu_count, total_urls)
        logger.info(f"Auto-detected workers: {num_workers} (CPU cores: {cpu_count}, files: {total_urls})")
    else:
        num_workers = min(args.workers, total_urls)
        if args.workers > total_urls:
            logger.info(f"Capping workers from {args.workers} to {num_workers} (number of files)")
    
    logger.info(f"Total files to download: {total_urls}")
    logger.info(f"Save directory: {args.save_dir}")
    logger.info(f"Workers: {num_workers}")
    logger.info(f"Stagger delay: {args.stagger_delay}s per worker")
    logger.info(f"Max retries per file: {args.max_retries}")
    logger.info(f"Base wait time: {args.base_wait}s (exponential backoff on errors)")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info("")
    logger.info("Download mode: Parallel with staggered worker starts")
    if num_workers > 1:
        for i in range(min(num_workers, 4)):  # Show first 4 workers as examples
            wait_time = args.stagger_delay * i
            if i == 0:
                logger.info(f"  - Worker 0: starts immediately")
            else:
                logger.info(f"  - Worker {i}: starts after {wait_time}s")
        if num_workers > 4:
            logger.info(f"  - ... ({num_workers - 4} more workers)")
    else:
        logger.info(f"  - Single worker mode (sequential downloads)")
    logger.info("Strategy: Browser-like headers + staggered starts + exponential backoff on 503 errors")
    logger.info("")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Prepare download tasks with worker indices
    download_tasks = []
    for idx, filename in enumerate(files_to_download):
        full_url = download_url_prefix + filename
        worker_idx = idx % num_workers  # Assign worker index in round-robin fashion
        download_tasks.append((full_url, args.save_dir, args.skip_existing, args.max_retries, args.base_wait, worker_idx, args.stagger_delay))
    
    # Statistics
    stats = {
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'error': 0
    }
    
    # Download with thread pool
    logger.info(f"Starting parallel downloads with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all download tasks
        futures = {
            executor.submit(download_url, url, save_dir, skip, max_retries, base_wait, worker_idx, stagger_delay): url
            for url, save_dir, skip, max_retries, base_wait, worker_idx, stagger_delay in download_tasks
        }
        
        # Process completed downloads with progress bar
        with tqdm(total=total_urls, desc="Downloading", unit="file") as pbar:
            for future in as_completed(futures):
                result = future.result()
                stats[result['status']] += 1
                
                # Update progress bar with statistics
                pbar.set_postfix({
                    'OK': stats['success'],
                    'Skip': stats['skipped'],
                    'Fail': stats['failed'] + stats['error']
                })
                pbar.update(1)
    
    # Final summary
    logger.info("=" * 60)
    logger.info("Download Summary:")
    logger.info(f"  Total items:      {total_urls}")
    logger.info(f"  Successfully downloaded: {stats['success']}")
    logger.info(f"  Skipped (existing):      {stats['skipped']}")
    logger.info(f"  Failed:                  {stats['failed']}")
    logger.info(f"  Errors:                  {stats['error']}")
    logger.info("=" * 60)


