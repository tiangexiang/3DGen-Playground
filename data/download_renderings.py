import os
import json
import argparse
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

dwonload_url_prefix = 'https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/gobjaverse_alignment/'


def download_url(url, save_dir, skip_existing=True):
    """
    Download a file from URL to save_dir.
    
    Args:
        url: The URL to download from
        save_dir: Directory to save the file
        skip_existing: If True, skip download if file already exists
        
    Returns:
        dict: Status information about the download
    """
    try:
        # Parse URL to get directory structure and filename
        url_parts = url.split('/')
        object_id = url_parts[-2]  # e.g., the object folder name
        filename = url_parts[-1]   # e.g., the actual file
        
        # Create target directory
        target_dir = os.path.join(save_dir, object_id)
        os.makedirs(target_dir, exist_ok=True)
        
        # Full path to the file
        target_file = os.path.join(target_dir, filename)
        
        # Check if file already exists
        if skip_existing and os.path.exists(target_file):
            file_size = os.path.getsize(target_file)
            if file_size > 0:  # Only skip if file is not empty
                logger.debug(f"Skipping {object_id}/{filename} (already exists, {file_size} bytes)")
                return {
                    'status': 'skipped',
                    'url': url,
                    'object_id': object_id,
                    'filename': filename,
                    'reason': 'already_exists'
                }
        
        # Download using wget with retry and progress suppression for cleaner parallel output
        command = [
            'wget',
            '--tries=3',  # Retry up to 3 times
            '--timeout=30',  # 30 second timeout
            '--continue',  # Continue partial downloads
            '--no-verbose',  # Less verbose output
            '-O', target_file,  # Output file
            url
        ]
        
        logger.debug(f"Downloading {object_id}/{filename}...")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            file_size = os.path.getsize(target_file) if os.path.exists(target_file) else 0
            logger.info(f"✓ Downloaded {object_id}/{filename} ({file_size} bytes)")
            return {
                'status': 'success',
                'url': url,
                'object_id': object_id,
                'filename': filename,
                'size': file_size
            }
        else:
            logger.error(f"✗ Failed to download {object_id}/{filename}: {result.stderr}")
            return {
                'status': 'failed',
                'url': url,
                'object_id': object_id,
                'filename': filename,
                'error': result.stderr
            }
            
    except Exception as e:
        logger.error(f"✗ Error downloading {url}: {str(e)}")
        return {
            'status': 'error',
            'url': url,
            'error': str(e)
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download gobjaverse alignment renderings.')
    parser.add_argument('--json_path', type=str, required=True, default='gobjaverse_alignment.json', 
                        help='Path to JSON file containing URLs')
    parser.add_argument('--save_dir', type=str, required=True, default='save_dir/', 
                        help='Directory to save downloaded files')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), 
                        help=f'Number of parallel download workers (default: {os.cpu_count()} - CPU cores)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip files that already exist (default: True)')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                        help='Re-download all files even if they exist')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose debug logging')

    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load URLs from JSON
    logger.info(f"Loading URLs from {args.json_path}")
    with open(args.json_path, 'r') as f:
        urls_dict = json.load(f)
    
    total_urls = len(urls_dict)
    logger.info(f"Found {total_urls} items to download")
    logger.info(f"Save directory: {args.save_dir}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Skip existing: {args.skip_existing}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Prepare download tasks
    download_tasks = []
    for key, url_path in urls_dict.items():
        full_url = os.path.join(dwonload_url_prefix, url_path)
        download_tasks.append((full_url, args.save_dir, args.skip_existing))
    
    # Statistics
    stats = {
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'error': 0
    }
    
    # Download with thread pool
    logger.info(f"Starting downloads...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all download tasks
        futures = {
            executor.submit(download_url, url, save_dir, skip): (url, save_dir)
            for url, save_dir, skip in download_tasks
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


