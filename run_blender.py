#!/usr/bin/env python3
"""
Blender Launcher Script

Automatically detects Blender installation and runs scripts in headless mode.
Works across Windows, macOS, and Linux.

Usage:
    python run_blender.py --script generate_human.py -- --config config.json
    python run_blender.py --script generate_human.py -- --config config.json --rig-type default_no_toes
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
import platform


class BlenderFinder:
    """Utility class to find Blender installations across different operating systems."""

    # Common Blender installation paths by OS
    SEARCH_PATHS = {
        'Windows': [
            # Standard installation paths
            r'C:\Program Files\Blender Foundation',
            r'C:\Program Files (x86)\Blender Foundation',
            # User-specific installation
            Path.home() / 'AppData' / 'Local' / 'Programs' / 'Blender Foundation',
            # Portable installations
            r'C:\Blender',
            Path.home() / 'Blender',
        ],
        'Darwin': [  # macOS
            '/Applications',
            Path.home() / 'Applications',
            '/opt/homebrew/bin',
            '/usr/local/bin',
        ],
        'Linux': [
            '/usr/bin',
            '/usr/local/bin',
            '/snap/bin',
            '/opt/blender',
            Path.home() / '.local' / 'bin',
            Path.home() / 'blender',
        ]
    }

    def __init__(self, config_file='.blender_config.json'):
        self.config_file = Path(config_file)
        self.os_name = platform.system()
        self.cached_path = None

    def load_cached_path(self):
        """Load previously found Blender path from config file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    path = config.get('blender_path')
                    if path and Path(path).exists():
                        print(f"Using cached Blender path: {path}")
                        return path
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def save_path(self, path):
        """Save Blender path to config file for future use."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump({'blender_path': str(path)}, f, indent=2)
            print(f"Saved Blender path to {self.config_file}")
        except IOError as e:
            print(f"Warning: Could not save Blender path: {e}")

    def find_blender_windows(self):
        """Find Blender executable on Windows."""
        search_paths = [Path(p) for p in self.SEARCH_PATHS['Windows']]

        for base_path in search_paths:
            if not base_path.exists():
                continue

            # Look for Blender version folders
            if 'Blender Foundation' in str(base_path):
                # Search in version subdirectories
                for version_dir in sorted(base_path.glob('Blender*'), reverse=True):
                    blender_exe = version_dir / 'blender.exe'
                    if blender_exe.exists():
                        return str(blender_exe)
            else:
                # Direct search for blender.exe
                blender_exe = base_path / 'blender.exe'
                if blender_exe.exists():
                    return str(blender_exe)

        return None

    def find_blender_macos(self):
        """Find Blender executable on macOS."""
        search_paths = [Path(p) for p in self.SEARCH_PATHS['Darwin']]

        for base_path in search_paths:
            if not base_path.exists():
                continue

            # Look for Blender.app
            if base_path.name in ['Applications']:
                for app_dir in sorted(base_path.glob('Blender*.app'), reverse=True):
                    blender_exe = app_dir / 'Contents' / 'MacOS' / 'Blender'
                    if blender_exe.exists():
                        return str(blender_exe)
            else:
                # Direct blender binary
                blender_exe = base_path / 'blender'
                if blender_exe.exists() and os.access(blender_exe, os.X_OK):
                    return str(blender_exe)

        return None

    def find_blender_linux(self):
        """Find Blender executable on Linux."""
        search_paths = [Path(p) for p in self.SEARCH_PATHS['Linux']]

        for base_path in search_paths:
            if not base_path.exists():
                continue

            # Look for blender executable
            blender_exe = base_path / 'blender'
            if blender_exe.exists() and os.access(blender_exe, os.X_OK):
                return str(blender_exe)

            # Look in version subdirectories
            for version_dir in sorted(base_path.glob('blender-*'), reverse=True):
                blender_exe = version_dir / 'blender'
                if blender_exe.exists() and os.access(blender_exe, os.X_OK):
                    return str(blender_exe)

        return None

    def find_in_path(self):
        """Check if blender is available in system PATH."""
        try:
            result = subprocess.run(
                ['blender', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Try to get full path
                if self.os_name == 'Windows':
                    where_result = subprocess.run(
                        ['where', 'blender'],
                        capture_output=True,
                        text=True
                    )
                    if where_result.returncode == 0:
                        return where_result.stdout.strip().split('\n')[0]
                else:
                    which_result = subprocess.run(
                        ['which', 'blender'],
                        capture_output=True,
                        text=True
                    )
                    if which_result.returncode == 0:
                        return which_result.stdout.strip()
                return 'blender'  # Available in PATH
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def find_blender(self):
        """
        Find Blender executable across different operating systems.

        Returns:
            str: Path to Blender executable, or None if not found
        """
        # Check cache first
        cached = self.load_cached_path()
        if cached:
            return cached

        # Check environment variable
        env_path = os.environ.get('BLENDER_PATH')
        if env_path and Path(env_path).exists():
            print(f"Using Blender from BLENDER_PATH: {env_path}")
            self.save_path(env_path)
            return env_path

        # Check system PATH
        print("Checking system PATH...")
        path_result = self.find_in_path()
        if path_result:
            print(f"Found Blender in PATH: {path_result}")
            self.save_path(path_result)
            return path_result

        # Search common installation directories
        print(f"Searching common installation directories for {self.os_name}...")

        if self.os_name == 'Windows':
            blender_path = self.find_blender_windows()
        elif self.os_name == 'Darwin':
            blender_path = self.find_blender_macos()
        elif self.os_name == 'Linux':
            blender_path = self.find_blender_linux()
        else:
            print(f"Warning: Unsupported operating system: {self.os_name}")
            blender_path = None

        if blender_path:
            print(f"Found Blender at: {blender_path}")
            self.save_path(blender_path)
            return blender_path

        return None

    def get_blender_version(self, blender_path):
        """Get Blender version information."""
        try:
            result = subprocess.run(
                [blender_path, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse version from output
                version_line = result.stdout.strip().split('\n')[0]
                return version_line
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return "Unknown version"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Blender scripts in headless mode with automatic Blender detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run generate_human.py with config
    python run_blender.py --script generate_human.py -- --config config.json

    # Run with specific rig type
    python run_blender.py --script generate_human.py -- --config config.json --rig-type default_no_toes

    # Manually specify Blender path
    python run_blender.py --blender-path "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe" --script generate_human.py -- --config config.json

    # Just find and display Blender path
    python run_blender.py --find-only
        """
    )

    parser.add_argument(
        '--script',
        type=str,
        help='Python script to run in Blender'
    )

    parser.add_argument(
        '--blender-path',
        type=str,
        help='Manual path to Blender executable (overrides auto-detection)'
    )

    parser.add_argument(
        '--find-only',
        action='store_true',
        help='Only find and display Blender path, do not run script'
    )

    parser.add_argument(
        '--gui',
        action='store_true',
        help='Run Blender with GUI instead of background mode'
    )

    # All arguments after '--' are passed to the Blender script
    args, script_args = parser.parse_known_args()

    # Remove the first '--' if it's in script_args (argparse keeps it)
    if script_args and script_args[0] == '--':
        script_args = script_args[1:]

    return args, script_args


def main():
    """Main execution function."""
    args, script_args = parse_arguments()

    print("\n" + "="*70)
    print("BLENDER LAUNCHER")
    print("="*70 + "\n")

    # Initialize Blender finder
    finder = BlenderFinder()

    # Find Blender executable
    if args.blender_path:
        blender_path = args.blender_path
        if not Path(blender_path).exists():
            print(f"Error: Specified Blender path does not exist: {blender_path}")
            return 1
        print(f"Using manually specified Blender path: {blender_path}")
    else:
        blender_path = finder.find_blender()
        if not blender_path:
            print("\n ERROR: Could not find Blender installation!")
            print("\nPlease either:")
            print("  1. Install Blender from https://www.blender.org/download/")
            print("  2. Set BLENDER_PATH environment variable to your Blender executable")
            print("  3. Use --blender-path argument to specify path manually")
            print(f"\nSearched in:")
            for path in finder.SEARCH_PATHS.get(finder.os_name, []):
                print(f"  - {path}")
            return 1

    # Get version info
    version = finder.get_blender_version(blender_path)
    print(f"Blender version: {version}")

    # If find-only mode, exit here
    if args.find_only:
        print(f"\n Blender executable: {blender_path}")
        return 0

    # Validate script argument
    if not args.script:
        print("\n ERROR: --script argument is required")
        print("Usage: python run_blender.py --script generate_human.py -- --config config.json")
        return 1

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"\n ERROR: Script file not found: {script_path}")
        return 1

    # Build Blender command
    blender_cmd = [blender_path]

    if not args.gui:
        blender_cmd.append('--background')

    blender_cmd.extend(['--python', str(script_path.absolute())])

    # Add script arguments if any
    if script_args:
        blender_cmd.append('--')
        blender_cmd.extend(script_args)

    # Display command
    print("\n" + "-"*70)
    print("Executing command:")
    print("-"*70)
    print(' '.join(blender_cmd))
    print("-"*70 + "\n")

    # Run Blender with script
    try:
        result = subprocess.run(blender_cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n Error running Blender: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
